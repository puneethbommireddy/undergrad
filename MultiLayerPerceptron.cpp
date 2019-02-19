#include <iostream>
#include <armadillo>
#include <vector>
#include <random>


using namespace std;
using namespace arma;

enum ActFun {hsig,htan};
enum Task {Regress,Classify};

vec sigm(vec a)
{
	return 1/(1 + exp(-a));
}

class Parameters
{
public:
	vector<int>SL;
	vector<ActFun> h;
	int Nl;
	Task task;
};

pair<field<vec>,field<vec>> FwdPrp(field<mat> W,vec x,Parameters Net)
{
	field<vec> A(Net.Nl),Z(Net.Nl);
	vec u = {1};
	for(int layNum = 0; layNum < Net.Nl; layNum++)
	{
		if(layNum == 0)
			A(0) = W(0)*join_vert(u,x);
		else
		{
			switch(Net.h[layNum])
			{
				case htan:	Z(layNum-1) = join_vert(u,tanh(A(layNum-1)));
							break;
				case hsig:	Z(layNum-1) = join_vert(u,sigm(A(layNum-1)));
							break;
			}
			A(layNum) = W(layNum)*Z(layNum-1);//here
		}
	}
	if(Net.task == Classify)
		Z(Net.Nl-1) = normalise(exp(A(Net.Nl-1)));
	return pair<field<vec>,field<vec>>(A,Z);
}

field<mat> BckPrp(field<mat> W,vector<vec> X,vector<vec> T,Parameters Net)
{
	vec dh;
	field<vec> A,Z;
	int paus;
	float Err,eps=0.5,lda=1e-1,eta = 7e-3/X.size();
	vec u = {1};
	int iter = 0;
	field<mat> dW(Net.Nl),zW(Net.Nl);
	for(int layNum = 1;layNum <= Net.Nl;layNum++)
		zW(layNum-1) = zeros<mat>(Net.SL[layNum],Net.SL[layNum-1]+1);
	field<vec> del(Net.Nl);
	do
	{
		dW = zW;
		Err = 0;iter++;
		for(int egN = 0; egN < X.size(); egN++)
		{	
			pair<field<vec>,field<vec>> res = FwdPrp(W,X[egN],Net);
			A = res.first;	Z = res.second;
			for(int layNum = Net.Nl-1; layNum >= 0; layNum--)
			{
				if(layNum == Net.Nl-1)
				{
					switch(Net.task)
					{
						case Regress:	del(Net.Nl-1) = A(Net.Nl-1) - T[egN];//here
										break;
						//case Classify:	del(Net.Nl-1) = T[egN], Z(Net.Nl-1));
					}
				}
				else
				{
					switch(Net.h[layNum])
					{
						case htan:	dh = 1 - tanh(A(layNum)) % tanh(A(layNum));
									break;
						case hsig:	dh = sigm(A(layNum)) % (1 - sigm(A(layNum)));
									break;
					}
					del(layNum) = dh % ( W(layNum+1).cols(1,W(layNum+1).n_cols-1).t() * del(layNum+1) );//W(dim)(out,in)
				}
				if (layNum == 0)
					dW(layNum) = dW(layNum) - kron(del(layNum),join_vert(u,X[egN]).t());
				else
					dW(layNum) = dW(layNum) - kron(del(layNum),Z(layNum-1).t());
			}
			Err = Err + sum(del(Net.Nl-1) % del(Net.Nl-1));
		}
		for(int layNum = 0; layNum < Net.Nl; layNum++)
			W(layNum) = W(layNum) + eta*dW(layNum);
		Err = sqrt(Err/X.size());
		cout<<Err<<"\n";
	} while (iter < 100000);
	W.print();	
	return W;
}

int main(int argc, char** argv)
{
	cout.precision(10);
	cout.setf(ios::fixed);
	
	vec x = {1,0},r,theta;
	vec t = {0};
	vector<vec> X{x}; vector<vec> T{t};
	for(int i = 0;i<100;i++)
	{
		theta = 2 * datum::pi * randu<vec>(1);
		r = 25*randu<vec>(1) + 5;
		X.push_back(join_vert(r,theta));
		T.push_back( cos(datum::pi/2.0 * cos(theta))/(sin(theta)*r)  );
	}
		
	Parameters Net;
	Net.SL = {1,5,1};
	Net.Nl = Net.SL.size() - 1;
	Net.h = {hsig,hsig,hsig};
	
	field<mat> W(Net.Nl);
	for(int layNum = 1;layNum <= Net.Nl;layNum++)
		W(layNum-1) = sqrt(6.0/(Net.SL[layNum]+Net.SL[layNum-1]+1)) * randu<mat>(Net.SL[layNum],Net.SL[layNum-1]+1);
	
	W.load("sqr.mat");
	//W = BckPrp(W,X,T,Net);
	//W.print("W=");
	//W.save("hwave15.mat");

	x = {8.5};
	pair<field<vec>,field<vec>> res = FwdPrp(W,x,Net);
	cout<<"Y"<<vectorise(res.first(Net.Nl-1))<<"\n";
	
	return 0;
}