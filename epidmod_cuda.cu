#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <vector>
#include <algorithm>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/normal_distribution.h>

__global__ void init(
	int *state,
	bool *isUpdate,
	float *chance,
	thrust::minstd_rand *gen,
	thrust::uniform_real_distribution<float> *randFloat,
	thrust::uniform_int_distribution<int> *randPeople,
	thrust::normal_distribution<float> *randRemoved,
	thrust::normal_distribution<float> *randInfected,
	unsigned int *connAddr,
	unsigned int *connSize,
	unsigned int *conns,
	int peopleAmount,
	int totalThreads,
	unsigned int connOneSide,
	float infPart,
	float expPart,
	time_t seed
){
	int startIndex = threadIdx.x + blockIdx.x * blockDim.x;
	for(int i = startIndex; i < peopleAmount; i += totalThreads)
	{
		gen[i] = thrust::minstd_rand(seed + i * 10000);
		randFloat[i] = thrust::uniform_real_distribution<float>(0, 1);
		randPeople[i] = thrust::uniform_int_distribution<int>(0, peopleAmount - 1);
		randRemoved[i] = thrust::normal_distribution<float>(13.5f, 6.0f);
		randInfected[i] = thrust::normal_distribution<float>(5.5f, 3.0f);
		isUpdate[i] = false;

		// init states
		state[i] = 0;
		chance[i] = (randFloat[i])(gen[i]);
		if (i < (int)((expPart + infPart) * peopleAmount)) 
		{    // exposed
			(state[i])++;
			chance[i] = (randInfected[i])(gen[i]);
		}
		if (i < (int)(infPart * peopleAmount)) 
		{                                // infectious
			(state[i])++;
			chance[i] = (randRemoved[i])(gen[i]);
		}

		// init graph
		connAddr[i] = i * connOneSide * 2;
		connSize[i] = connOneSide * 2;
		for(int j = 0; j < connOneSide; j++)
		{
			conns[connAddr[i] + j] = (i + j + 1) % peopleAmount;
			conns[connAddr[i] + connOneSide + j] = (i - j - 1 + peopleAmount) % peopleAmount;
		}
	}
}

__global__ void calcUpdStates(
	int *state,
	bool *isUpdate,
	float *chance,
	thrust::minstd_rand *gen,
	thrust::uniform_int_distribution<int> *randPeople,
	unsigned int *connAddr,
	unsigned int *connSize,
	unsigned int *conns,
	unsigned int *randConns,
	int randConnsAmount,
	int peopleAmount,
	int totalThreads,
	float infChance
){
	int startIndex = threadIdx.x + blockIdx.x * blockDim.x;
	for(int i = startIndex; i < peopleAmount; i += totalThreads)
	{// calc chances
		if(state[i] == 0)
		{
			for(unsigned int rndInd = i * randConnsAmount; rndInd < (i + 1) * randConnsAmount; rndInd++)
			{// add random edges
				randConns[rndInd] = (randPeople[i])(gen[i]);
			}

			//calc chance
			float ch = 1.0f;
			for(unsigned int ind = connAddr[i]; ind < connAddr[i] + connSize[i]; ind++)
			{
				if(state[conns[ind]] == 2) { ch *= 1 - infChance; }
			}
			for(unsigned int rndInd = i * randConnsAmount; rndInd < (i + 1) * randConnsAmount; rndInd++)
			{
				bool isAdd = true;
				for(unsigned int ind = connAddr[i]; ind < connAddr[i] + connSize[i]; ind++)
				{
					if(conns[ind] == randConns[rndInd]) { isAdd = false; break; }
				}
				if(isAdd)
				{
					for(unsigned int j = i * randConnsAmount; j < rndInd; j++)
					{
						if(randConns[j] == randConns[rndInd]) { isAdd = false; break; }
					}

					if(isAdd)
					{
						if(state[randConns[rndInd]] == 1) { ch *= 1 - expChance; }
						if(state[randConns[rndInd]] == 2) { ch *= 1 - infChance; }
					}
				}
			}
			//check chance
			if(ch < chance[i]) { isUpdate[i] = true; }

		}

		if(state[i] == 1 || state[i] == 2)
		{
			chance[i]--;
			if(chance[i] <= 0) { isUpdate[i] = true; }
		}

	}
}


__global__ void updateStates(
	int *state,
	bool *isUpdate,
	float *chance,
	thrust::minstd_rand *gen,
	thrust::uniform_real_distribution<float> *randFloat,
	thrust::normal_distribution<float> *randRemoved,
	thrust::normal_distribution<float> *randInfected,
	int peopleAmount,
	int totalThreads
){
	int startIndex = threadIdx.x + blockIdx.x * blockDim.x;
	for(int i = startIndex; i < peopleAmount; i += totalThreads)
	{
		if(isUpdate[i])
		{
			if(state[i] == 0) { chance[i] = (randInfected[i])(gen[i]); }
			if(state[i] == 1) { chance[i] = (randRemoved[i])(gen[i]); }
			state[i]++;
			isUpdate[i] = false;
		}

		if(state[i] == 0) { chance[i] = (randFloat[i])(gen[i]); }
	}
}



int main()
{

	printf("watts-strogaz reroll with vectors, cuda\n");
	// declaring variables , do changes here vvv
	int threads = 64;
	int blocks = 64;

	unsigned int peopleAmount = 300000;

	unsigned int connAmount = 6;
	unsigned int randConnsAmount = 6;

	float infectedPart = 0.05f;
	float exposedPart = 0.1f;

	float wsgk = 0.2f;

	float infCatch = 0.0558f;

	unsigned int repeatTime = 50; // days

	unsigned int repeats = 500; // loops for time counting
	// no more changes from here ^^^^

	int totalThreads = threads * blocks;

	unsigned int connOneSide = connAmount / 2;

	thrust::minstd_rand *devGen;
	thrust::uniform_real_distribution<float> *devRandFloat;
	thrust::uniform_int_distribution<int> *devRandPeople;
	thrust::normal_distribution<float> *devRandRemoved;
	thrust::normal_distribution<float> *devRandInfected;

	int *state, *devState;
	bool *devIsUpdate;
	float *devChance;
	unsigned int *connAddr,*devConnAddr, *connSize, *devConnSize, *conns, *devConns;
	unsigned int *devRandConns;

	// memory allocating
	connAddr = (unsigned int*)malloc(sizeof(unsigned int) * peopleAmount);
	connSize = (unsigned int*)malloc(sizeof(unsigned int) * peopleAmount);
	conns = (unsigned int*)malloc(sizeof(unsigned int) * peopleAmount * connOneSide * 2);
	cudaMalloc((void**)&devConnAddr, sizeof(unsigned int) * peopleAmount);
	cudaMalloc((void**)&devConnSize, sizeof(unsigned int) * peopleAmount);
	cudaMalloc((void**)&devConns, sizeof(unsigned int) * peopleAmount * connOneSide * 2);

	cudaMalloc((void**)&devRandConns, sizeof(unsigned int) * peopleAmount * randConnsAmount);

	state = (int*)malloc(sizeof(int) * peopleAmount);
	cudaMalloc((void**)&devState, sizeof(int) * peopleAmount);
	cudaMalloc((void**)&devIsUpdate, sizeof(bool) * peopleAmount);
	cudaMalloc((void**)&devChance, sizeof(float) * peopleAmount);
	cudaMalloc((void**)&devGen, sizeof(thrust::minstd_rand) * peopleAmount);
	cudaMalloc((void**)&devRandFloat, sizeof(thrust::uniform_real_distribution<float>) * peopleAmount);
	cudaMalloc((void**)&devRandPeople, sizeof(thrust::uniform_int_distribution<int>) * peopleAmount);
	cudaMalloc((void**)&devRandRemoved, sizeof(thrust::normal_distribution<float>) * peopleAmount);
	cudaMalloc((void**)&devRandInfected, sizeof(thrust::normal_distribution<float>) * peopleAmount);

	// do things
	time_t begin = time(0);
	printf("people: %u\n", peopleAmount);
	for(unsigned int n = 0; n < repeats; n++)
	{

		time_t timeSeed = time(0);
		// initialize variables
		init<<<blocks,threads>>>(
			devState,
			devIsUpdate,
			devChance,
			devGen,
			devRandFloat,
			devRandPeople,
			devRandRemoved,
			devRandInfected,
			devConnAddr,
			devConnSize,
			devConns,
			peopleAmount,
			totalThreads,
			connOneSide,
			infectedPart,
			exposedPart,
			timeSeed
		);

		// watts-strogaz reroll
		cudaMemcpy(connAddr, devConnAddr, sizeof(unsigned int) * peopleAmount, cudaMemcpyDeviceToHost);
		cudaMemcpy(connSize, devConnSize, sizeof(unsigned int) * peopleAmount, cudaMemcpyDeviceToHost);
		cudaMemcpy(conns, devConns, sizeof(unsigned int) * peopleAmount * connOneSide * 2, cudaMemcpyDeviceToHost);

		thrust::minstd_rand gen(timeSeed - 10000);
		thrust::uniform_real_distribution<float> randFloat(0, 1);
		thrust::uniform_int_distribution<int> randPeople(0, peopleAmount - 1);

		// vector realisation of watts-strogatz
		std::vector<unsigned int>  vecConns[peopleAmount];
		for(unsigned int i = 0; i < peopleAmount; i++)
		{
			std::vector<unsigned int> toAdd = std::vector<unsigned int>();
			for(unsigned int j = connAddr[i]; j < connAddr[i] + connSize[i]; j++) { toAdd.push_back(conns[j]); }
			vecConns[i] = toAdd;
		}

		for(unsigned int i = 0; i < peopleAmount; i++)
		{
			for (unsigned int ind = 0; ind < vecConns[i].size(); ind++)
			{
				if (randFloat(gen) < wsgk)
				{       // if edge need to be replaced
					unsigned int new_adr;
					do { new_adr = randPeople(gen); } while (new_adr == i);         // new edge without connecting vertex to itself
					if (std::find(vecConns[new_adr].begin(), vecConns[new_adr].end(), i) == vecConns[new_adr].end())
					{               // if edge not exist
						vecConns[new_adr].push_back(i);         // adding edge to 'new' vertex
						vecConns[vecConns[i][ind]].erase(std::find(vecConns[vecConns[i][ind]].begin(), vecConns[vecConns[i][ind]].end(), i));           // remove edge from 'old' vertex
						vecConns[i][ind] = new_adr;     // change edge to initial vertex
					}
				}
			}
		}
		unsigned int ind = 0;
		for(unsigned int i = 0; i < peopleAmount; i++)
		{
			connAddr[i] = ind;
			connSize[i] = vecConns[i].size();
			for(unsigned int j = 0; j < vecConns[i].size(); j++) { conns[ind] = vecConns[i][j]; }
		}
		cudaMemcpy(devConnAddr, connAddr, sizeof(unsigned int) * peopleAmount, cudaMemcpyHostToDevice);
		cudaMemcpy(devConnSize, connSize, sizeof(unsigned int) * peopleAmount, cudaMemcpyHostToDevice);
		cudaMemcpy(devConns, conns, sizeof(unsigned int) * peopleAmount * connOneSide * 2, cudaMemcpyHostToDevice);

		// end of watts-strogaz

		for(unsigned int t = 0; t < repeatTime; t++)
		{// updating states
			calcUpdStates<<<blocks,threads>>>(
				devState,
				devIsUpdate,
				devChance,
				devGen,
				devRandPeople,
				devConnAddr,
				devConnSize,
				devConns,
				devRandConns,
				randConnsAmount,
				peopleAmount,
				totalThreads,
				infCatch
			);

			updateStates<<<blocks,threads>>>(
				devState,
				devIsUpdate,
				devChance,
				devGen,
				devRandFloat,
				devRandRemoved,
				devRandInfected,
				peopleAmount,
				totalThreads
			);
		}

		cudaMemcpy(state, devState, sizeof(int) * peopleAmount, cudaMemcpyDeviceToHost);

		int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
		for(int i = 0; i < peopleAmount; i++) {
		if(state[i] == 0) { s0++; }
		if(state[i] == 1) { s1++; }
		if(state[i] == 2) { s2++; }
		if(state[i] == 3) { s3++; }
		}

		printf("%u:\t%i\t%i\t%i\t%i\n", n, s0, s1, s2, s3);
	}
	time_t end = time(0);
	printf("%llu\n%f\n", end - begin, static_cast<float>(end - begin) / static_cast<float>(repeats));

	// memory free
	cudaFree(devState);
	cudaFree(devIsUpdate);
	cudaFree(devChance);
	cudaFree(devGen);
	cudaFree(devRandFloat);
	cudaFree(devRandPeople);
	cudaFree(devRandRemoved);
	cudaFree(devRandInfected);
	cudaFree(devConnAddr);
	cudaFree(devConnSize);
	cudaFree(devConns);
	free(connAddr);
	free(connSize);
	free(conns);
	free(state);

	return(0);
}
