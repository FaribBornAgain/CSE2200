#include<bits/stdc++.h>

using namespace std;

// Jacobi Iterative Method
void jacobiMethod(const vector<vector<double>>& A, const vector<double>& b, vector<double>& x, int maxIterations = 100, double tolerance = 1e-10) {
    int n = A.size();
    vector<double> x_old(n);

    for (int k = 0; k < maxIterations; ++k) {
        x_old = x;
        for (int i = 0; i < n; ++i) {
            double sigma = 0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sigma += A[i][j] * x_old[j];
                }
            }
            x[i] = (b[i] - sigma) / A[i][i];
        }

        double error = 0;
        for (int i = 0; i < n; ++i) {
            error += abs(x[i] - x_old[i]);
        }
        if (error < tolerance) break;
    }
}

// Gauss-Seidel Iterative Method
void gaussSeidelMethod(const vector<vector<double>>& A, const vector<double>& b, vector<double>& x, int maxIterations = 100, double tolerance = 1e-10) {
    int n = A.size();

    for (int k = 0; k < maxIterations; ++k) {
        vector<double> x_old = x;
        for (int i = 0; i < n; ++i) {
            double sigma = 0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sigma += A[i][j] * x[j];
                }
            }
            x[i] = (b[i] - sigma) / A[i][i];
        }

        double error = 0;
        for (int i = 0; i < n; ++i) {
            error += abs(x[i] - x_old[i]);
        }
        if (error < tolerance) break;
    }
}

// Gauss Elimination
void gaussElimination(vector<vector<double>> A, vector<double> b, vector<double>& x) {
    int n = A.size();

    for (int i = 0; i < n; ++i) {
        for (int k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}

// Gauss-Jordan Elimination
void gaussJordanElimination(vector<vector<double>> A, vector<double> b, vector<double>& x) {
    int n = A.size();

    for (int i = 0; i < n; ++i) {
        double pivot = A[i][i];
        for (int j = 0; j < n; ++j) {
            A[i][j] /= pivot;
        }
        b[i] /= pivot;

        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = 0; j < n; ++j) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
    }

    x = b;
}

// LU Factorization
void luFactorization(const vector<vector<double>>& A, vector<double> b, vector<double>& x) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0)), U = A;

    for (int i = 0; i < n; ++i) {
        L[i][i] = 1;
        for (int j = i + 1; j < n; ++j) {
            L[j][i] = U[j][i] / U[i][i];
            for (int k = i; k < n; ++k) {
                U[j][k] -= L[j][i] * U[i][k];
            }
        }
    }

    vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= L[i][j] * y[j];
        }
    }

    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

void printSolution(const vector<double>& x) {
    for (int i = 0; i < x.size(); ++i) {
        cout << "x" << i + 1 << " = " << x[i] << endl;
    }
}

int main() {
    int n;
    cout << "Enter the number of equations (size of the system): ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n), x(n, 0);

    cout << "Enter the coefficients of matrix A:" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> A[i][j];
        }
    }

    cout << "Enter the constants of vector b:" << endl;
    for (int i = 0; i < n; ++i) {
        cin >> b[i];
    }

    cout << "\nJacobi Iterative Method:\n";
    jacobiMethod(A, b, x);
    printSolution(x);

    x.assign(n, 0);
    cout << "\nGauss-Seidel Iterative Method:\n";
    gaussSeidelMethod(A, b, x);
    printSolution(x);

    x.resize(n);
    cout << "\nGauss Elimination:\n";
    gaussElimination(A, b, x);
    printSolution(x);

    x.resize(n);
    cout << "\nGauss-Jordan Elimination:\n";
    gaussJordanElimination(A, b, x);
    printSolution(x);

    x.resize(n);
    cout << "\nLU Factorization:\n";
    luFactorization(A, b, x);
    printSolution(x);

    return 0;
}
/*
4 1 2
3 5 1
1 1 2
4
7
3
*/
