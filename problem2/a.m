clc; clear; close all;

vr = 10;           % radial vel
vtheta = 3;        % angular vel
vtheta_rad = vtheta * pi/180;  
T = 200;           % total time
initial_pos = [1000, 0];  % initial pos

% one point per second
t = 0:1:T;

% position arrays
x = zeros(size(t));
y = zeros(size(t));

% initial position
x(1) = initial_pos(1);
y(1) = initial_pos(2);

% trajectory calc
for i = 2:length(t)
    % current radius and angle
    r_prev = sqrt(x(i-1)^2 + y(i-1)^2);
    theta_prev = atan2(y(i-1), x(i-1));
    
    % Update 
    r_new = r_prev + vr;  
    theta_new = theta_prev + vtheta_rad;  
    
    % Cartesian coordinates
    x(i) = r_new * cos(theta_new);
    y(i) = r_new * sin(theta_new);
end

% Plot the trajectory
figure;
plot(x, y, 'b-');
hold on;
plot(x(1), y(1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');  % start point
plot(0, 0, 'kx', 'MarkerSize', 10);  % Origin: Madhuri's position
grid on;
xlabel('x (meters)');
ylabel('y (meters)');
title('(a) Trajectory of Siva''s Car');
legend('Trajectory', 'Starting Position', 'Origin (Madhuri''s Position)');
axis equal; 