#include <iostream>
#include <vector>
#include <armadillo>
#include <limits>

using namespace std;
using namespace arma;

vec rSimplex(mat A, vec b, vec c, vec x, uvec dash = zeros<uvec>(0))
{
	int n = A.n_cols, m = A.n_rows,k = -1, j = 0;
	uvec idx = zeros<uvec>(n);
	if (dash.n_rows == 0)
	{
		dash = zeros<uvec>(m);
		vec d = A*x - b;
		for (int i = 0; i < m; i++) if(d(i) == 0) { dash(i) = 1; idx(j++) = i;}
	}
	else
	{
		for (int i = 0; i < m; i++) if(dash(i) == 1) idx(j++) = i;
		x = inv(A.rows(idx))*b(idx);
	}
	mat Z = -inv(A.rows(idx));vec ic = (c.t()*Z).t();
	float t = numeric_limits<float>::infinity();
	for (int i = 0; i < n; i++)
	{
		if(ic(i) > 0)
		{
			for (int s = 0; s < m; s++)
			{
				if (dash(s) == 0)
				{
					float den = dot(A.row(s).t(),Z.col(i));
					float num = b(s) - dot(A.row(s).t(),x);
					if (den > 0 && t > num/den)
					{
						t = num/den;
						k = s;
					}
				}
			}
			dash(idx(i)) = 0; dash(k) = 1;
			break;
		}
	}
	if (k != -1) return rSimplex(A,b,c,x,dash);
	else return inv(A.rows(idx))*b(idx);
}

int main()
{
	mat A;
	A << -1 << 0 << endr
	  << 0 << -1 << endr
	  << 2 << -1 << endr
	  << 1 << -5 << endr
	vec c{2,-1}, b{0,0,2,-4},x{-4,0};
	vec ans = rSimplex(A,b,c,x);
	ans.print();
	return 0;
}