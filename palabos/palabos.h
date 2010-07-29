#ifndef _H_PALABOS
#define _H_PALABOS

struct D3Q19_DistF
{
	float dist[19];
	bool mask;
	void *unused;
};

struct D3Q15_DistF
{
	float dist[15];
	bool mask;
	void *unused;
};

struct D3Q13_DistF
{
	float dist[13];
	bool mask;
	void *unused;
};

struct D2Q9_DistF
{
	float dist[9];
	bool mask;
	void *unused;
};

struct BGKParams
{
	float tau;
};

struct MRT_D2Q9_Params
{
	float tau[9];
};

struct MRT_D3Q13_Params
{
	float tau[13];
};

struct MRT_D3Q15_Params
{
	float tau[15];
};

struct MRT_D3Q19_Params
{
	float tau[19];
};

#endif /* _H_PALABOS */
