#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

typedef struct {
	float x;
	float y;
	float z;
} vec3;

class Body {
public:
	bool blank;
	vec3 position;
	vec3 velocity;
	float mass;
	float radius;
	uint8_t* color;
	Body();
	Body(vec3 position, float mass, float radius);
	Body(vec3 position, vec3 velocity, float mass, float radius);
	void setColor(uint8_t r, uint8_t g, uint8_t b);
};

Body::Body() {
	this->blank = true;
}

Body::Body(vec3 position, float mass, float radius) {
	vec3 vel;
	vel.x = 0;
	vel.y = 0;
	vel.z = 0;

	this->position = position;
	this->velocity = vel;
	this->mass = mass;
	this->radius = radius;
	this->color = new uint8_t[3]{ 200, 0, 200 };
	this->blank = false;
}

Body::Body(vec3 position, vec3 velocity, float mass, float radius) {
	this->position = position;
	this->velocity = velocity;
	this->mass = mass;
	this->radius = radius;
	this->color = new uint8_t[3]{ 255, 0, 255 };
	this->blank = false;
}

void Body::setColor(uint8_t r, uint8_t g, uint8_t b) {
	this->color = new uint8_t[3]{ r, g, b };
}

__global__
void step(Body* bodiesIn, Body* results, int n, float dt) {
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
		Body a = bodiesIn[i];
		float EPS = 3e4;
		float G = 1.0;
		float fx = 0;
		float fy = 0;
		float fz = 0;

		for (int j = 0; j < n; j++) {
			if (j != i) {
				Body b = bodiesIn[j];

				float dx = b.position.x - a.position.x;
				float dy = b.position.y - a.position.y;
				float dz = b.position.z - a.position.z;
				float dist = sqrt(dx*dx + dy*dy + dz*dz);

				float F = (G * a.mass * b.mass) / (dist * dist + EPS * EPS);
				fx += F * dx / dist;
				fy += F * dy / dist;
				fz += F * dz / dist;
			}
		}

		a.velocity.x += dt * fx / a.mass;
		a.velocity.y += dt * fy / a.mass;
		a.velocity.z += dt * fz / a.mass;
		a.position.x += dt * a.velocity.x;
		a.position.y += dt * a.velocity.y;
		a.position.z += dt * a.velocity.z;

		results[i] = a;
	}
	
}

__global__
void prepareForNextStep(Body* bodiesIn, Body* results, int n) {
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
		bodiesIn[i] = results[i];
	}
}


class Simulation {
public:
	Simulation(unsigned int maxBodies, const int numThreads);
	void addBody(Body b);
	void addBody(vec3 position, vec3 velocity, float mass, float radius);
	void sendBodiesToDevice();
	void readBodiesFromDevice();
	void stepSimulation(float dt);
	Body getBody(int index);
	void cleanup();
	void enableFrameRecord();

private:
	unsigned int currentStep;
	unsigned int maxBodies;
	unsigned int numBodies;
	int numThreads;
	int numSMs;
	bool recordFrames;
	Body* bodies;
	Body* inBodies;
	Body* resBodies;
	std::ofstream recordFile;
	void recordFrame();
};

Simulation::Simulation(unsigned int maxBodies, const int numThreads) {
	this->maxBodies = maxBodies;
	this->bodies = new Body[maxBodies];
	this->numBodies = 0;
	this->currentStep = 0;
	this->numThreads = numThreads;
	this->recordFrames = false;
	this->numSMs = 0;

	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
}

void Simulation::addBody(Body b) {
	if (numBodies < maxBodies) {
		bodies[numBodies] = b;
		numBodies++;
	}
	
}

void Simulation::addBody(vec3 position, vec3 velocity, float mass, float radius) {
	if (numBodies < maxBodies) {
		bodies[numBodies] = Body(position, velocity, mass, radius);
		numBodies++;
	}
}

void Simulation::stepSimulation(float dt) {
	if (currentStep == 0) {
		step<<< 32 * numSMs, numThreads >>>(inBodies, resBodies, numBodies, 1.0);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		currentStep++;
	}
	else {
		prepareForNextStep<<< 32*numSMs, numThreads >>>(inBodies, resBodies, numBodies);
		step<<< 32*numSMs, numThreads >>>(inBodies, resBodies, numBodies, 1.0);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		currentStep++;
	}
	if(recordFrames){
		recordFrame();
	}
}

// Only need to do this once
// Copy bodies from bodies array to gpu, and allocate space for result bodies on gpu
void Simulation::sendBodiesToDevice() {
	cudaMalloc(&inBodies, sizeof(Body) * maxBodies);
	cudaMalloc(&resBodies, sizeof(Body) * maxBodies);
	cudaMemcpy(inBodies, bodies, sizeof(Body) * maxBodies, cudaMemcpyHostToDevice);
}

// Copy current simulation step from gpu to bodies array
void Simulation::readBodiesFromDevice() {
	cudaMemcpy(bodies, resBodies, sizeof(Body) * maxBodies, cudaMemcpyDeviceToHost);
}

Body Simulation::getBody(int index) {
	return bodies[index];
}

void Simulation::cleanup() {
	cudaFree(inBodies);

	//delete[] bodies;
	//delete[] resBodies;

}

void Simulation::enableFrameRecord(){
	int err = system("rm -r frames");
	if (-1 == err)
	{
    	printf("Error creating directory!n");
    	exit(1);
	}

	err = system("mkdir -p frames");
	if (-1 == err)
	{
    	printf("Error creating directory!n");
    	exit(1);
	}

	recordFrames = true;
}

void Simulation::recordFrame(){
	readBodiesFromDevice();
	std::stringstream num;
	num << "frames/" << currentStep << ".bdat";
	recordFile.open(num.str().c_str());

	if(recordFile.is_open()){
	for(int l = 0; l < maxBodies; l++){
		Body b = this->getBody(l);
		float px = b.position.x;
		float py = b.position.y;
		float pz = b.position.z;
		recordFile << l << " " << px << " " << py << " " << pz << " " << "\n";
	}
	}
	recordFile.flush();
	recordFile.close();
}

int main(int argc, char** argv) {
	int maxBodies = 1000;
	int threads = 256;
	int cycles = 1000;
	bool record = false;

	if(argc >= 4){
	std::cout << "Starting simulation with " << atoi(argv[1]) << " bodies and " << atoi(argv[2]) << " threads" << std::endl;
		maxBodies = atoi(argv[1]);
		threads = atoi(argv[2]);
		cycles = atoi(argv[3]);
	}
	else{
		std::cout << "No parameters given, starting with 1000 bodies running on 256 threads" << std::endl;
	}

	if(argc == 5 && atoi(argv[4]) == 1){
		std::cout << "Enabled data recording, frame data will be stored in the 'frames' directory" << std::endl;
		record = true;
	}

	Simulation* sim = new Simulation(maxBodies, threads);
	if(record){
	sim -> enableFrameRecord();
	}
	srand(time(NULL));

	for (int i = 0; i < maxBodies; i++) {
		vec3 pos;
		vec3 vel;

		pos.x = rand() % 10000;
		pos.y = rand() % 10000;
		pos.z = rand() % 10000;

		vel.x = 0;
		vel.y = 0;
		vel.z = 0;
		
		sim->addBody(pos, vel, 1000000 - rand() % 1000, 100);
	}

	std::cout << "Bodies added, moving to gpu memory" << std::endl;
	std::cout << "6th body: ";

	std::cout << std::fixed;
	std::cout << std::setprecision(2);

	Body c = sim->getBody(5);
	std::cout << " x-" << c.position.x << " y-" << c.position.y << " z-" << c.position.z << std::endl;

	sim->sendBodiesToDevice();

	std::cout << "Gpu memory populated, stepping simulation " << cycles << " times" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < cycles; i++) {
		sim->stepSimulation(1.0);
	}
	auto finish = std::chrono::high_resolution_clock::now();
	long long elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	double elapsed_seconds = elapsed / 1000000000.0;
	long long rate = (long long)(cycles * maxBodies) / elapsed_seconds;
	long long totalBodies = cycles * maxBodies;

	std::cout << cycles << " cycles done, " << elapsed_seconds << " seconds elapsed, reading results" << std::endl;

	sim->readBodiesFromDevice();

	std::cout << "6th result: ";

	Body b = sim->getBody(5);
	vec3 bpos = b.position;

	std::cout << "x-" << bpos.x << " y-" << bpos.y << " z-" << bpos.z << std::endl;
	std::cout << totalBodies << " bodies processed (total over " << cycles << " cycles) in " << elapsed_seconds << " seconds (" << (long)rate << " bodies per second)" << std::endl;

	sim->cleanup();
	return 0;
}