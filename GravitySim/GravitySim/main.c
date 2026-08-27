#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

const char* vertexShaderSource =
	"#version 330 core\n"
	"layout(location=0) in vec3 aPos;\n"
	"uniform mat4 model;\n"
	"uniform mat4 view;\n"
	"uniform mat4 projection;\n"
	"out float lightIntensity;\n"
	"\n"
	"void main() {\n"
	"	gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
	"	vec3 worldPos = (model * vec4(aPos, 1.0)).xyz;\n"
	"	vec3 normal = normalize(aPos);\n"
	"	vec3 dirToCenter = normalize(-worldPos);\n"
	"	lightIntensity = max(dot(normal, dirToCenter), 0.15);\n"
	"}\n";

const char* fragmentShaderSource =
	"#version 330 core\n"
	"in float lightIntensity;\n"
	"out vec4 FragColor;\n"
	"uniform vec4 objectColor;\n"
	"uniform bool isGrid;\n"
	"uniform bool glow;\n"
	"\n"
	"void main() {\n"
	"	if (isGrid) {\n"
	"		FragColor = objectColor;\n"
	"	} else if (glow) {\n"
	"		FragColor = vec4(objectColor.rgb * 100000, objectColor.a);\n"
	"	} else {\n"
	"		float fade = smoothstep(0.0, 10.0, lightIntensity * 10);\n"
	"		FragColor = vec4(objectColor.rgb * fade, objectColor.a);\n"
	"	}\n"
	"}\n";

typedef struct Object {
	GLuint VAO;
	GLuint VBO;

	vec3 position;
	vec3 velocity;
	vec3 lastPos;
	vec4 color;

	size_t vertexCount;

	bool isInitializing;
	bool isLaunched;
	bool isTarget;
	bool isGlow;

	float mass;
	float density;
	float radius;
} Object;

#define MAX_OBJECTS 10000

Object objects[MAX_OBJECTS];

GLuint gridVAO = 0;
GLuint gridVBO = 0;

bool isRunning = true;
bool isPaused = true;
vec3 cameraPosition = { 0.0f, 0.0f, 1.0f };
vec3 cameraFront = { 0.0f, 0.0f, -1.0f };
vec3 cameraUp = { 0.0f, 1.0f, 0.0f };
float lastX = 400.0f;
float lastY = 300.0f;
float yaw = -90.0f;
float pitch = 0.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
size_t objectCount = 0;

const double G = 6.6743e-11; // m^3 kg^-1 s^-2
const float c = 299792458.0f; // m/s
const double initMass = 1e22;
float sizeRatio = 30000.0f;

GLFWwindow* startWindow();
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource);
void createVBOVAO(GLuint* VAO, GLuint* VBO, const float* vertices, size_t vertexCount);
void updateCamera(GLuint shaderProgram, vec3 cameraPosition);
void keyCallback(GLFWwindow* window, int button, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mouseCallback(GLFWwindow* window, double xPos, double yPos);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
void sphericalToCartesian(float r, float theta, float phi, vec3 out) {
	out[0] = r * sinf(theta) * cosf(phi); // x
	out[1] = r * cosf(theta);             // y
	out[2] = r * sinf(theta) * sinf(phi); // z
}
void drawGrid(GLuint shaderProgram, GLuint gridVAO, size_t vertexCount);
float* createGridVertices(float size, int divisions, const Object* objects, size_t objectCount, size_t outVertexCount);
float* updateGridVertices(float* vertices, size_t vertexCount, const Object* objects, size_t objectCount);

float* objectDraw(const Object* object, size_t* outVertexCount) {
	const int stacks = 10;
	const int sectors = 10;

	/* Each stack/sector produces:
	 * 2 triangles
	 * 3 vertices
	 * 3 floats
	 * There are stacks * sectors patches
	 *
	 * 6 vertices per quad
	 * 3 floats per vertex
	 */

	size_t vertexCount = (size_t)stacks * sectors * 6;
	size_t floatCount = vertexCount * 3;

	float* vertices = malloc(floatCount * sizeof(float));

	if (vertices == NULL) {
		*outVertexCount = 0;
		return NULL;
	}

	size_t index = 0;

	for (int i = 0; i < stacks; ++i) {
		float theta1 = ((float)i / stacks) * (float)M_PI;
		float theta2 = ((float)(i + 1) / stacks) * (float)M_PI;

		for (int j = 0; j < sectors; ++j) {
			float phi1 = ((float)j / sectors) * 2.0f * (float)M_PI;
			float phi2 = ((float)(j + 1) / sectors) * 2.0f * (float)M_PI;

			vec3 v1, v2, v3, v4;

			sphericalToCartesian(object->radius, theta1, phi1, v1);
			sphericalToCartesian(object->radius, theta1, phi2, v2);
			sphericalToCartesian(object->radius, theta2, phi1, v3);
			sphericalToCartesian(object->radius, theta2, phi2, v4);

			// Triangle 1: v1-v2-v3
			vertices[index++] = v1[0];
			vertices[index++] = v1[1];
			vertices[index++] = v1[2];

			vertices[index++] = v2[0];
			vertices[index++] = v2[1];
			vertices[index++] = v2[2];

			vertices[index++] = v3[0];
			vertices[index++] = v3[1];
			vertices[index++] = v3[2];

			// Triangle 2: v2-v4-v3
			vertices[index++] = v2[0];
			vertices[index++] = v2[1];
			vertices[index++] = v2[2];

			vertices[index++] = v4[0];
			vertices[index++] = v4[1];
			vertices[index++] = v4[2];

			vertices[index++] = v3[0];
			vertices[index++] = v3[1];
			vertices[index++] = v3[2];
		}
	}

	*outVertexCount = vertexCount;

	return vertices;
}

void objectInit(Object *object, vec3 initPosition, vec3 initVelocity, float mass, float density, vec4 color, bool glow) {
	glm_vec3_copy(initPosition, object->position);
	glm_vec3_copy(initVelocity, object->velocity);

	object->mass = mass;
	object->density = density;
	object->radius = powf((3.0f * object->mass / object->density) / (4.0f * (float)M_PI), 1.0f / 3.0f) / sizeRatio;

	glm_vec4_copy(color, object->color);

	object->lastPos[0] = object->position[0];
	object->lastPos[1] = object->position[1];
	object->lastPos[2] = object->position[2];

	object->isInitializing = false;
	object->isLaunched = false;
	object->isTarget = false;
	object->isGlow = glow;

	// Generate sphere vertices
	size_t vertexCount = 0;
	float* vertices = objectDraw(object, &vertexCount);

	object->vertexCount = vertexCount;

	createVBOVAO(&object->VAO, &object->VBO, vertices, vertexCount);

	free(vertices);
}

//void updateObjectPosition(Object* object) {
//	object->position[0] += object->velocity[0] / 94.0f;
//	object->position[1] += object->velocity[1] / 94.0f;
//	object->position[2] += object->velocity[2] / 94.0f;
//
//	object->radius = powf((3.0f * object->mass / object->density) / (4.0f * (float)M_PI), 1.0f / 3.0f) / sizeRatio;
//}

void updateObjectPosition(Object* object) {
	vec3 displacement;

	glm_vec3_scale(object->velocity, 1.0f * 94.0f, displacement);

	glm_vec3_add(object->position, displacement, object->position);

	object->radius = powf((3.0f * object->mass / object->density) / (4.0f * (float)M_PI), 1.0f / 3.0f) / sizeRatio;
}

void updateObjectVertices(Object* object) {
	size_t vertexCount;
	float* vertices = objectDraw(object, &vertexCount);

	if (vertices == NULL) {
		return;
	}

	glBindBuffer(GL_ARRAY_BUFFER, object->VBO);

	glBufferData(GL_ARRAY_BUFFER, vertexCount * 3 * sizeof(float), vertices, GL_STATIC_DRAW);

	object->vertexCount = vertexCount;

	free(vertices);
}

void objectAccelerate(Object* object, float x, float y, float z) {
	object->velocity[0] += x / 94.0f;
	object->velocity[1] += y / 94.0f;
	object->velocity[2] += z / 94.0f;
}

float checkObjectCollision(const Object* object, const Object* other) {
	float dx = other->position[0] - object->position[0];
	float dy = other->position[1] - object->position[1];
	float dz = other->position[2] - object->position[2];

	float distance = sqrtf(dx * dx + dy * dy + dz * dz);

	if (other->radius + object->radius > distance) {
		return -0.2f;
	}

	return 1.0f;
}

int main() {
	GLFWwindow* window = startWindow();
	GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);

	GLint modelLocation = glGetUniformLocation(shaderProgram, "model");
	GLint objectColorLocation = glGetUniformLocation(shaderProgram, "objectColor");
	GLint projectionLocation = glGetUniformLocation(shaderProgram, "projection");
	GLint isGridLocation = glGetUniformLocation(shaderProgram, "isGrid");
	GLint glowLocation = glGetUniformLocation(shaderProgram, "glow");

	glUseProgram(shaderProgram);

	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Matrix projection
	mat4 projection;

	glm_perspective(glm_rad(45.0f), 800.0f / 600.0f, 0.1f, 750000.0f, projection);
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat *)projection);

	// Camera position
	glm_vec3_copy((vec3) { 0.0f, 1000.0f, 5000.0f }, cameraPosition);

	objectInit(&objects[objectCount++], (vec3) { -5000.0f, 650.0f, -350.0f }, (vec3) { 0.0f, 0.0f, 1500.0f }, 5.97219e22, 5515.0f, (vec4){ 0.0f, 1.0f, 1.0f, 1.0f }, false);
	objectInit(&objects[objectCount++], (vec3) { 5000.0f, 650.0f, -350.0f }, (vec3) { 0.0f, 0.0f, -1500.0f }, 5.97219e22, 5515.0f, (vec4) { 0.0f, 1.0f, 1.0f, 1.0f }, false);
	objectInit(&objects[objectCount++], (vec3) { 0.0f, 0.0f, -350.0f }, (vec3) { 0.0f, 0.0f, 0.0f }, 1.989e25, 5515.0f, (vec4) { 1.0f, 0.929f, 0.176f, 1.0f }, false);

	// Vertices
	size_t gridVertexCount = 0;

	float* gridVertices = createGridVertices(20000.0f, 25, objects, objectCount, &gridVertexCount);

	createVBOVAO(&gridVAO, &gridVBO, gridVertices, gridVertexCount);

	while (!glfwWindowShouldClose(window) && isRunning) {
		float currentFrame = (float)glfwGetTime();

		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		updateCamera(shaderProgram, cameraPosition);

		// Increase mass while initialising
		if (objectCount > 0 && objects[objectCount - 1].isInitializing) {
			Object* object = &objects[objectCount - 1];

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
				// Increase mass by 1% per second
				object->mass *= 1.0 + 1.0 * deltaTime;

				// Update radius
				object->radius = (float)(pow((3.0 * object->mass / object->density) / (4.0 * M_PI), 1.0 / 3.0) / sizeRatio);

				updateObjectVertices(object);
			}
		}

		// Draw grid
		glUseProgram(shaderProgram);

		glUniform4f(objectColorLocation, 1.0f, 1.0f, 1.0f, 0.25f);
		glUniform1i(isGridLocation, 1);
		glUniform1i(glowLocation, 0);

		// updateGridVertices() returns new array and update vertex count
		//float* newGridVertices = updateGridVertices(gridVertices, &gridVertexCount, objects, objectCount);

		//if (newGridVertices != gridVertices) {
		//	free(gridVertices);
		//	gridVertices = newGridVertices;
		//}
		updateGridVertices(gridVertices, gridVertexCount, objects, objectCount);

		glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
		glBufferData(GL_ARRAY_BUFFER, gridVertexCount * sizeof(float), gridVertices, GL_DYNAMIC_DRAW);
		drawGrid(shaderProgram, gridVAO, gridVertexCount);

		// Draw objects
		for (size_t i = 0; i < objectCount; ++i) {
			Object* object = &objects[i];

			glUniform4f(objectColorLocation, object->color[0], object->color[1], object->color[2], object->color[3]);

			// Calculate the gravitational interaction with every other object
			for (size_t j = 0; j < objectCount; ++j) {
				Object* object2 = &objects[j];

				if (object2 != object && !object->isInitializing && !object2->isInitializing) {
					float dx = object2->position[0] - object->position[0];
					float dy = object2->position[1] - object->position[1];
					float dz = object2->position[2] - object->position[2];
					float distance = sqrtf(dx * dx + dy * dy + dz * dz);

					if (distance > 0.0f) {
						// direction vector
						vec3 direction = { dx / distance, dy / distance, dz / distance };

						distance *= 1000.0f;

						double Gforce = (G * object->mass * object2->mass) / (distance * distance);

						float acceleration1 = (float)(Gforce / object->mass);
						vec3 acceleration = { direction[0] * acceleration1, direction[1] * acceleration1, direction[2] * acceleration1 };

						if (!isPaused) {
							objectAccelerate(object, acceleration[0], acceleration[1], acceleration[2]);

							// Collision
							float collisionFactor = checkObjectCollision(object, object2);
							glm_vec3_scale(object->velocity, collisionFactor, object->velocity);
							printf("radius: %f\n", object->radius);
						}
					}
				}
			}

			// Initialising object radius
			if (object->isInitializing) {
				object->radius = (float)(pow((3.0 * object->mass / object->density) / (4.0 * M_PI), 1.0 / 3.0) / sizeRatio);
				updateObjectVertices(object);
			}

			if (!isPaused) {
				updateObjectPosition(object);
			}

			// Model matrix
			mat4 model;

			glm_mat4_identity(model);

			glm_translate(model, object->position);

			glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)model);

			glUniform1i(isGridLocation, 0);

			glUniform1i(glowLocation, object->isGlow ? 1 : 0);

			// Draw sphere
			glBindVertexArray(object->VAO);

			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(object->vertexCount / 3));
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup objects
	for (size_t i = 0; i < objectCount; ++i) {
		glDeleteVertexArrays(1, &objects[i].VAO);
		glDeleteBuffers(1, &objects[i].VBO);
	}

	glDeleteVertexArrays(1, &gridVAO);
	glDeleteBuffers(1, &gridVBO);

	glDeleteProgram(shaderProgram);

	free(gridVertices);

	glfwTerminate();

	return 0;
}

GLFWwindow* startWindow() {
	if (!glfwInit()) {
		printf("Failed to initialise GLFW");
		return NULL;
	}

	GLFWwindow* window = glfwCreateWindow(800, 600, "3D Test", NULL, NULL);

	if (!window) {
		printf("Failed to create GLFW window");
		glfwTerminate();
		return NULL;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;

	if (glewInit() != GLEW_OK) {
		printf("Failed to initialise GLEW");
		glfwTerminate();
		return NULL;
	}

	glEnable(GL_DEPTH_TEST);
	glViewport(0, 0, 800, 600);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	return window;
}

GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
	// Vertex shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);

	GLint success;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		printf("Vertex shader compilation failed: %s", infoLog);
	}

	// Fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		printf("Fragment shader compilation failed: %s", infoLog);
	}

	// Shader program
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		printf("Shader program linking failed: %s", infoLog);
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

void createVBOVAO(GLuint* VAO, GLuint *VBO, const float* vertices, size_t vertexCount) {
	glGenVertexArrays(1, VAO);
	glGenBuffers(1, VBO);

	glBindVertexArray(*VAO);
	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(float), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}

void updateCamera(GLuint shaderProgram, vec3 cameraPosition) {
	glUseProgram(shaderProgram);
	
	vec3 target;
	glm_vec3_add(cameraPosition, cameraFront, target);

	mat4 view;
	glm_lookat(cameraPosition, target, cameraUp, view);

	GLint viewLocation = glGetUniformLocation(shaderProgram, "view");

	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (float*)view);
}

void keyCallback(GLFWwindow* window, int key, int action, int mods) {
	float cameraSpeed = 10000.0f * deltaTime;
	bool shiftPressed = (mods & GLFW_MOD_SHIFT) != 0;

	// Camera movement
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		glm_vec3_muladds(cameraFront, cameraSpeed, cameraPosition);
	}

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		glm_vec3_muladds(cameraFront, -cameraSpeed, cameraPosition);
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		vec3 right;
		glm_vec3_cross(cameraFront, cameraUp, right);
		glm_vec3_normalize(right);
		glm_vec3_muladds(right, -cameraSpeed, cameraPosition);
	}

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		vec3 right;
		glm_vec3_cross(cameraFront, cameraUp, right);
		glm_vec3_normalize(right);
		glm_vec3_muladds(right, cameraSpeed, cameraPosition);
	}

	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		glm_vec3_muladds(cameraUp, cameraSpeed, cameraPosition);
	}

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		glm_vec3_muladds(cameraUp, -cameraSpeed, cameraPosition);
	}

	// Pause
	if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
		isPaused = true;
	}

	if (glfwGetKey(window, GLFW_KEY_K) == GLFW_RELEASE) {
		isPaused = false;
	}

	// Quit
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
		isRunning = false;
	}

	// Don't try to access objects[-1]
	if (objectCount == 0) {
		return;
	}

	// Get the last object
	Object* lastObject = &objects[objectCount - 1];

	// Only allow movement while the object is initializing
	if (lastObject->isInitializing) {
		if (key == GLFW_KEY_UP && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
			if (!shiftPressed) {
				lastObject->position[1] += lastObject->radius * 0.2f;
			} else {
				lastObject->position[2] += lastObject->radius * 0.2f;
			}
		}

		if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
			if (!shiftPressed) {
				lastObject->position[1] -= lastObject->radius * 0.2f;
			} else {
				lastObject->position[2] -= lastObject->radius * 0.2f;
			}
		}

		if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
			lastObject->position[0] += lastObject->radius * 0.2f;
		}

		if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
			lastObject->position[0] -= lastObject->radius * 0.2f;
		}
	}
}

void mouseCallback(GLFWwindow* window, double xPos, double yPos) {
	float xOffset = xPos - lastX;
	float yOffset = lastY - yPos;
	lastX = xPos;
	lastY = yPos;

	float sensitivity = 0.1f;
	xOffset *= sensitivity;
	yOffset *= sensitivity;

	yaw += xOffset;
	pitch += yOffset;

	if (pitch > 89.0f) pitch = 89.0f;
	if (pitch < -89.0f) pitch = -89.09f;

	float yawRad = glm_rad(yaw);
	float pitchRad = glm_rad(pitch);

	vec3 front = { cosf(yawRad) * cosf(pitchRad),
				   sinf(pitchRad),
				   sinf(yawRad) * cosf(pitchRad)
	};

	glm_vec3_normalize(front);
	glm_vec3_copy(front, cameraFront);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			if (objectCount < MAX_OBJECTS) {
				objectInit(&objects[objectCount], 
						   (vec3) { 0.0f, 0.0f, 0.0f }, 
						   (vec3) { 0.0f, 0.0f, 0.0f }, 
						   initMass, 
						   5515.0f, 
						   (vec4) { 1.0f, 1.0f, 1.0f, 1.0f }, 
						   false
				);

				objects[objectCount].isInitializing = true;
				objectCount++;
			}
		}

		if (action == GLFW_RELEASE && objectCount > 0) {
			Object* lastObject = &objects[objectCount - 1];

			lastObject->isInitializing = false;
			lastObject->isLaunched = true;
		}
	}

	if (objectCount > 0 && button == GLFW_MOUSE_BUTTON_RIGHT && objects[objectCount - 1].isInitializing) {
		Object* lastObject = &objects[objectCount - 1];

		if (action == GLFW_PRESS || action == GLFW_REPEAT) {
			lastObject->mass *= 1.2f;
			printf("Mass: %f\n", lastObject->mass);
		}
	}
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
	float cameraSpeed = 250000.0f * deltaTime;
	if (yOffset > 0) {
		glm_vec3_muladds(cameraFront, cameraSpeed, cameraPosition);
	} else if (yOffset < 0) {
		glm_vec3_muladds(cameraFront, -cameraSpeed, cameraPosition);
	}
}

void drawGrid(GLuint shaderProgram, GLuint gridVAO, size_t vertexCount) {
	glUseProgram(shaderProgram);
	
	mat4 model;
	glm_mat4_identity(model);

	GLint modelLocation = glGetUniformLocation(shaderProgram, "model");
	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)model);

	glBindVertexArray(gridVAO);
	glPointSize(5.0f);
	//glDrawArrays(GL_LINES, 0, (GLsizei)(vertexCount / 3));
	glDrawArrays(GL_LINES, 0, (GLsizei)vertexCount);
	glBindVertexArray(0);
}

//float* createGridVertices(float size, int divisions, const Object* objects, size_t objectCount, size_t* outVertexCount) {
//	float step = size / divisions;
//	float halfSize = size / 2.0f;
//
//	/* Number of vertices
//	 * X axis: (divisions + 1) * divisions * 2
//	 * Y axis: (divisions + 1) * divisions * 2
//	 */
//	//size_t vertexCount = ((size_t)(divisions + 1) * divisions * 2) + ((size_t)(divisions + 1) * divisions * 2);
//	size_t vertexCount = ((size_t)divisions * divisions * 2) + ((size_t)(divisions + 1) * divisions * 2);
//
//	// Each vertex has 3 floats
//	size_t floatCount = vertexCount * 3;
//
//	float* vertices = malloc(floatCount * sizeof(float));
//
//	if (vertices == NULL) {
//		*outVertexCount = 0;
//		return NULL;
//	}
//
//	size_t index = 0;
//
//	// X axis
//	for (int yStep = 3; yStep <= 3; ++yStep) {
//		float y = -halfSize * 0.3f + yStep * step;
//
//		for (int zStep = 0; zStep < divisions; ++zStep) {
//			float z = -halfSize + zStep * step;
//
//			for (int xStep = 0; xStep < divisions; ++xStep) {
//				float xStart = -halfSize + xStep * step;
//				float xEnd = xStart + step;
//
//				vertices[index++] = xStart;
//				vertices[index++] = y;
//				vertices[index++] = z;
//
//				vertices[index++] = xEnd;
//				vertices[index++] = y;
//				vertices[index++] = z;
//			}
//		}
//	}
//
//	// Z axis
//	for (int xStep = 0; xStep <= divisions; ++xStep) {
//		float x = -halfSize + xStep * step;
//
//		for (int yStep = 3; yStep <= 3; ++yStep) {
//			float y = -halfSize * 0.3f + yStep * step;
//
//			for (int zStep = 0; zStep < divisions; ++zStep) {
//				float zStart = -halfSize + zStep * step;
//				float zEnd = zStart + step;
//
//				vertices[index++] = x;
//				vertices[index++] = y;
//				vertices[index++] = zStart;
//
//				vertices[index++] = x;
//				vertices[index++] = y;
//				vertices[index++] = zEnd;
//			}
//		}
//	}
//
//	*outVertexCount = vertexCount;
//
//	return vertices;
//}

float* createGridVertices(float size, int divisions, const Object* objects, size_t objectCount, size_t* outVertexCount) {
	(void)objects;
	(void)objectCount;
	
	float step = size / divisions;
	float halfSize = size / 2.0f;

	/* Number of vertices
	 * X axis: divisions * divisions * 2 vertices
	 * Y axis: (divisions + 1) * divisions * 2 vertices
	 */
	 //size_t vertexCount = ((size_t)(divisions + 1) * divisions * 2) + ((size_t)(divisions + 1) * divisions * 2);
	size_t vertexCount = ((size_t)divisions * divisions * 2) + ((size_t)(divisions + 1) * divisions * 2);

	// Each vertex has 3 floats
	size_t floatCount = vertexCount * 3;

	float* vertices = malloc(floatCount * sizeof(float));

	if (vertices == NULL) {
		*outVertexCount = 0;
		return NULL;
	}

	size_t index = 0;

	// X axis
	for (int yStep = 3; yStep <= 3; ++yStep) {
		float y = -halfSize * 0.3f + yStep * step;

		for (int zStep = 0; zStep < divisions; ++zStep) {
			float z = -halfSize + zStep * step;

			for (int xStep = 0; xStep < divisions; ++xStep) {
				float xStart = -halfSize + xStep * step;
				float xEnd = xStart + step;

				vertices[index++] = xStart;
				vertices[index++] = y;
				vertices[index++] = z;

				vertices[index++] = xEnd;
				vertices[index++] = y;
				vertices[index++] = z;
			}
		}
	}

	// Z axis
	for (int xStep = 0; xStep <= divisions; ++xStep) {
		float x = -halfSize + xStep * step;

		for (int yStep = 3; yStep <= 3; ++yStep) {
			float y = -halfSize * 0.3f + yStep * step;

			for (int zStep = 0; zStep < divisions; ++zStep) {
				float zStart = -halfSize + zStep * step;
				float zEnd = zStart + step;

				vertices[index++] = x;
				vertices[index++] = y;
				vertices[index++] = zStart;

				vertices[index++] = x;
				vertices[index++] = y;
				vertices[index++] = zEnd;
			}
		}
	}

	*outVertexCount = index / 3;

	return vertices;
}

float* updateGridVertices(float* vertices, size_t vertexCount, const Object* objects, size_t objectCount) {
	// Calculate cente of mass
	float totalMass = 0.0f;
	float comY = 0.0f;

	for (size_t i = 0; i < objectCount; ++i) {
		const Object* object = &objects[i];

		if (object->isInitializing) {
			continue;
		}

		comY += object->mass * object->position[1];
		totalMass += object->mass;
	}

	if (totalMass > 0.0f) {
		comY /= totalMass;
	}

	// Find original maximum Y
	float originalMaxY = -INFINITY;

	for (size_t i = 0; i < vertexCount; i += 3) {
		if (vertices[i + 1] > originalMaxY) {
			originalMaxY = vertices[i + 1];
		}
	}

	float verticalShift = comY - originalMaxY;
	printf("Vertical shift: %f | comY: %f | originalMaxY: %f\n", verticalShift, comY, originalMaxY);

	// Bend space around objects
	for (size_t i = 0; i < vertexCount; i += 3) {
		float vertexX = vertices[i];
		float vertexY = vertices[i + 1];
		float vertexZ = vertices[i + 2];

		float totalDisplacementY = 0.0f;

		for (size_t j = 0; j < objectCount; ++j) {
			const Object* object = &objects[j];

			float toObjectX = object->position[0] - vertexX;
			float toObjectY = object->position[1] - vertexY;
			float toObjectZ = object->position[2] - vertexZ;

			float distance = sqrtf(toObjectX * toObjectX + toObjectY * toObjectY + toObjectZ * toObjectZ);

			float distanceM = distance * 1000.0f;

			float rs = (2.0f * (float)G * object->mass) / (c * c);

			float value = rs * (distanceM - rs);

			// prevent sqrtf() from receiving a negative value
			if (value > 0.0f) {
				float dz = 2.0f * sqrtf(value);
				totalDisplacementY += dz * 2.0f;
			}
		}

		vertices[i + 1] = totalDisplacementY - fabsf(verticalShift);
	}

	return vertices;
}