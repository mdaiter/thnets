#ifndef THNETWORK_H
#define THNETWORK_H

#ifndef TH_TENSOR_INC
#define TH_TENSOR_INC

typedef struct THFloatStorage
{
    float *data;
	int nref, mustfree;	// mustfree = 0 (allocated somewhere else), 1 (free), 2 (cuda free)
} THFloatStorage;

typedef struct THFloatTensor
{
    long size[4];
    long stride[4];
    int nDimension;    
	THFloatStorage *storage;
	long storageOffset;
#ifdef LOWP
	float sub, mult;
#endif
} THFloatTensor;
#endif

typedef struct thnetwork
{
	struct thobject *netobj;
	struct thobject *statobj;
	struct network *net;
	THFloatTensor *out;
	float mean[3], std[3];
	int grayscale;
} THNETWORK;
#endif
