clear all; close all; clc;
% Please do not change the template code before this line.

% Please complete the following with your details
firstname ='Zhengrui'; % the first name shown in your KEATS records
surname   ='Zhang'; % the surname shown in your KEATS records
number    ='k21022953'; % this should be your 'k' number, e.g., 'k1234567'

% Please complete Question 1 here.
L  = [20]; % Inductance
c  = [0.1]; % Capacitance
R  = [4]; % Resistance
%Take the state variables as x1=I(t), x2=dI(t)/dt, x3=d2I(t)/d2t, u=dV/dt.
%The state space expression is created from the differential equation, and
%then substitute the coefficients of each term into the vector matrix(Ac.*[x1;x2]+Bc.*u).
Ac = [0 1;-1/(L*c) -R/L]; % Continuous time A matrix
Bc = [0;1/L]; % Continuous time B matrix

% Please complete Question 2 here.
dt = [0.002]; % Sampling rate.
T  = [20]; % Simulation duration.
%The state space equations are discretized using forward differences, 
%and the transformation gives (x(k+1)-x(k))/T = Ac*x(k) + Bc*u(k).
%By simplifying it, we will find A = I + AcT, B = BcTds.(I is the identity matrix)
A  = [eye(2,2)+Ac*dt]; % Discrete time A matrix.
B  = [Bc*dt]; % Discrete time B matrix.
u  = [1]; % Control input.
x  = [0;0]; % Initial state.
S  = [T/dt];  % Number of simulation steps.
X  = zeros(2,S); % Matrix for storing data
for s = 1:S
    x = A*x + B*u;
    X(:,s) = x;
end

% Please complete Question 3 here.
%The output of the state space equation is the current.
C = [1,0]; % Observer ('C') matrix
H = [C;C*A]; % Observability matrix
%Use eigenvalue to find poles. Assume that the eigenvalue z, subtract it from the values 
%on the diagonal of matrix A respectively, and let the determinant of the changed matrix 
%equal to zero, then use the 'solve' function to calculate the value of z, which would be the poles.  
syms z
eq = (A(1)-z)*(A(4)-z)-(A(2)*A(3));
z = round(double(solve(eq==0)),4); % Vector containing poles in order of increasing size.

% Please DO NOT change the template code after this line.
save(['rlc_',number,'_',firstname,'_',surname]);
