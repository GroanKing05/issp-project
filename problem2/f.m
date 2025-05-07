% single sensor vs dual sensors
clear all; close all; clc;

% Load sensor data
load('range1.mat');
load('angle1.mat');
load('range2.mat');
load('angle2.mat');

angle1_rad = deg2rad(angle1);
angle2_rad = deg2rad(angle2);
dt = 1;

%% Calculate True Path
vr = 10;           % radial vel
vtheta = 3;        % angular vel
vtheta_rad = vtheta * pi/180;  
T = 200;           % total time
initial_pos = [1000, 0];  % initial pos

% one point per second
t = 0:1:T;

% position arrays
x_true = zeros(size(t));
y_true = zeros(size(t));

% initial position
x_true(1) = initial_pos(1);
y_true(1) = initial_pos(2);

% trajectory calc
for i = 2:length(t)
    % current radius and angle
    r_prev = sqrt(x_true(i-1)^2 + y_true(i-1)^2);
    theta_prev = atan2(y_true(i-1), x_true(i-1));
    
    % Update 
    r_new = r_prev + vr;  
    theta_new = theta_prev + vtheta_rad;  
    
    % Cartesian coordinates
    x_true(i) = r_new * cos(theta_new);
    y_true(i) = r_new * sin(theta_new);
end

% Calculate true state (r, theta, vr, vtheta) for each time point
r_true = sqrt(x_true.^2 + y_true.^2);
theta_true = atan2(y_true, x_true);
vr_true = ones(size(t)) * vr;
vtheta_true = ones(size(t)) * vtheta_rad;

%% Single Sensor Implementation
% Initial state and covariance
x_single = [500; 0; 0; 0]; % [r; theta; vr; vtheta]
P_single = diag([1000, 10, 50, 10]);

% Process noise covariance 
Q_w = zeros(4, 4);

% Measurement noise covariance
angle_var_rad = (deg2rad(4))^2; % Convert to radians^2
Q_v_single = diag([2500, angle_var_rad]);

% State transition matrix
A = [1, 0, dt, 0;
     0, 1, 0, dt;
     0, 0, 1, 0;
     0, 0, 0, 1];

% Observation matrix for single sensor
C_single = [1, 0, 0, 0;
            0, 1, 0, 0];

% Storage for results
x_history_single = zeros(4, length(range1));
x_history_single(:, 1) = x_single;
P_diag_history_single = zeros(4, length(range1));
P_diag_history_single(:, 1) = diag(P_single);

% Kalman filter loop - Single sensor
for n = 2:length(range1)
    % prediction
    x_pred = A * x_single;
    P_pred = A * P_single * A' + Q_w;
    
    % update
    y = [range1(n); angle1_rad(n)];
    y_diff = y - C_single * x_pred;
    S = C_single * P_pred * C_single' + Q_v_single;
    K = P_pred * C_single' / S; 
    
    x_single = x_pred + K * y_diff;
    P_single = (eye(4) - K * C_single) * P_pred;
    
    % Store results
    x_history_single(:, n) = x_single;
    P_diag_history_single(:, n) = diag(P_single);
end

%% Dual Sensor Implementation
% Initial state and covariance
x_dual = [500; 0; 0; 0]; % [r; theta; vr; vtheta]
P_dual = diag([1000, 10, 50, 10]);

% Measurement noise covariance for both sensor sets
Q_v_dual = diag([2500, angle_var_rad, 2500, angle_var_rad]);

% Observation matrix for both sensor sets
C_dual = [1, 0, 0, 0;
          0, 1, 0, 0;
          1, 0, 0, 0;
          0, 1, 0, 0];


x_history_dual = zeros(4, length(range1));
x_history_dual(:, 1) = x_dual;
P_diag_history_dual = zeros(4, length(range1));
P_diag_history_dual(:, 1) = diag(P_dual);

% Kalman filter loop - Dual sensors
for n = 2:length(range1)
    % Prediction 
    x_pred = A * x_dual;
    P_pred = A * P_dual * A' + Q_w;
    
    % update with both sensor sets
    y = [range1(n); angle1_rad(n); range2(n); angle2_rad(n)];
    y_diff = y - C_dual * x_pred;
    S = C_dual * P_pred * C_dual' + Q_v_dual;
    K = P_pred * C_dual' / S;
    
    x_dual = x_pred + K * y_diff;
    P_dual = (eye(4) - K * C_dual) * P_pred;
    
    % Store results
    x_history_dual(:, n) = x_dual;
    P_diag_history_dual(:, n) = diag(P_dual);
end

% Convert to Cartesian for plotting
x_cart_single = x_history_single(1,:) .* cos(x_history_single(2,:));
y_cart_single = x_history_single(1,:) .* sin(x_history_single(2,:));

x_cart_dual = x_history_dual(1,:) .* cos(x_history_dual(2,:));
y_cart_dual = x_history_dual(1,:) .* sin(x_history_dual(2,:));

% Error Calculation compared to true path
% For single sensor
error_r_single = abs(x_history_single(1,:) - r_true);
error_theta_single = abs(wrapToPi(x_history_single(2,:) - theta_true));
error_vr_single = abs(x_history_single(3,:) - vr_true);
error_vtheta_single = abs(x_history_single(4,:) - vtheta_true);

error_pos_single = sqrt((x_cart_single - x_true).^2 + (y_cart_single - y_true).^2);

% For dual sensors
error_r_dual = abs(x_history_dual(1,:) - r_true);
error_theta_dual = abs(wrapToPi(x_history_dual(2,:) - theta_true));
error_vr_dual = abs(x_history_dual(3,:) - vr_true);
error_vtheta_dual = abs(x_history_dual(4,:) - vtheta_true);

error_pos_dual = sqrt((x_cart_dual - x_true).^2 + (y_cart_dual - y_true).^2);

% Calculate RMS errors
rmse_pos_single = sqrt(mean(error_pos_single.^2));
rmse_r_single = sqrt(mean(error_r_single.^2));
rmse_theta_single = sqrt(mean(error_theta_single.^2));
rmse_vr_single = sqrt(mean(error_vr_single.^2));
rmse_vtheta_single = sqrt(mean(error_vtheta_single.^2));

rmse_pos_dual = sqrt(mean(error_pos_dual.^2));
rmse_r_dual = sqrt(mean(error_r_dual.^2));
rmse_theta_dual = sqrt(mean(error_theta_dual.^2));
rmse_vr_dual = sqrt(mean(error_vr_dual.^2));
rmse_vtheta_dual = sqrt(mean(error_vtheta_dual.^2));

% Calculate MAE
mae_pos_single = mean(error_pos_single);
mae_r_single = mean(error_r_single);
mae_theta_single = mean(error_theta_single);
mae_vr_single = mean(error_vr_single);
mae_vtheta_single = mean(error_vtheta_single);

mae_pos_dual = mean(error_pos_dual);
mae_r_dual = mean(error_r_dual);
mae_theta_dual = mean(error_theta_dual);
mae_vr_dual = mean(error_vr_dual);
mae_vtheta_dual = mean(error_vtheta_dual);

% Plot estimated trajectories with true path
figure;
plot(x_true, y_true, 'g-');
hold on;
plot(x_cart_single, y_cart_single, 'b-');
plot(x_cart_dual, y_cart_dual, 'r-');
plot(0, 0, 'ko', 'MarkerFaceColor', 'k');  % Origin (Madhuri's position)
xlabel('X position (m)');
ylabel('Y position (m)');
title('Comparison of Estimated Trajectories vs True Path');
legend('True Path', 'Single Sensor Set', 'Dual Sensor Sets', 'Origin (Madhuri''s Position)');
axis equal;
grid on;

% Plot position error over time
figure;
plot(t, error_pos_single, 'b-');
hold on;
plot(t, error_pos_dual, 'r-');
xlabel('Time (s)');
ylabel('Position Error (m)');
title('Position Error Over Time');
legend('Single Sensor', 'Dual Sensors');
grid on;

% Plot state errors
figure;
subplot(2,2,1);
plot(t, error_r_single, 'b-');
hold on;
plot(t, error_r_dual, 'r-');
xlabel('Time (s)');
ylabel('Range Error (m)');
title('Range Error vs Time');
legend('Single Sensor', 'Dual Sensors');
grid on;

subplot(2,2,2);
plot(t, rad2deg(error_theta_single), 'b-');
hold on;
plot(t, rad2deg(error_theta_dual), 'r-');
xlabel('Time (s)');
ylabel('Angle Error (deg)');
title('Angle Error vs Time');
legend('Single Sensor', 'Dual Sensors');
grid on;

subplot(2,2,3);
plot(t, error_vr_single, 'b-');
hold on;
plot(t, error_vr_dual, 'r-');
xlabel('Time (s)');
ylabel('Radial Velocity Error (m/s)');
title('Radial Velocity Error vs Time');
legend('Single Sensor', 'Dual Sensors');
grid on;

subplot(2,2,4);
plot(t, rad2deg(error_vtheta_single), 'b-');
hold on;
plot(t, rad2deg(error_vtheta_dual), 'r-');
xlabel('Time (s)');
ylabel('Angular Velocity Error (deg/s)');
title('Angular Velocity Error vs Time');
legend('Single Sensor', 'Dual Sensors');
grid on;

% Plot estimated velocities vs true values
figure;
subplot(2,1,1);
plot(t, vr_true, 'g-');
hold on;
plot(t, x_history_single(3,:), 'b-');
plot(t, x_history_dual(3,:), 'r-');
ylabel('Radial velocity (m/s)');
title('Comparison of Estimated vs True Radial Velocity');
legend('True Value', 'Single Sensor', 'Dual Sensors');
grid on;

subplot(2,1,2);
plot(t, rad2deg(vtheta_true), 'g-');
hold on;
plot(t, rad2deg(x_history_single(4,:)), 'b-');
plot(t, rad2deg(x_history_dual(4,:)), 'r-');
ylabel('Angular velocity (deg/s)');
xlabel('Time (s)');
title('Comparison of Estimated vs True Angular Velocity');
legend('True Value', 'Single Sensor', 'Dual Sensors');
grid on;

% Plot estimation uncertainty (diagonal elements of P)
figure;
subplot(2,2,1);
semilogy(t, P_diag_history_single(1,:), 'b-');
hold on;
semilogy(t, P_diag_history_dual(1,:), 'r-');
title('Range Estimation Uncertainty');
ylabel('Variance');
legend('Single Sensor', 'Dual Sensors');
grid on;

subplot(2,2,2);
semilogy(t, P_diag_history_single(2,:), 'b-');
hold on;
semilogy(t, P_diag_history_dual(2,:), 'r-');
title('Angle Estimation Uncertainty');
ylabel('Variance');
legend('Single Sensor', 'Dual Sensors');
grid on;

subplot(2,2,3);
semilogy(t, P_diag_history_single(3,:), 'b-');
hold on;
semilogy(t, P_diag_history_dual(3,:), 'r-');
title('Radial Velocity Estimation Uncertainty');
xlabel('Time (s)');
ylabel('Variance');
legend('Single Sensor', 'Dual Sensors');
grid on;

subplot(2,2,4);
semilogy(t, P_diag_history_single(4,:), 'b-');
hold on;
semilogy(t, P_diag_history_dual(4,:), 'r-');
title('Angular Velocity Estimation Uncertainty');
xlabel('Time (s)');
ylabel('Variance');
legend('Single Sensor', 'Dual Sensors');
grid on;

% Quantitative comparison
fprintf('=== Quantitative Metrics for Single and Dual Sensor Performance ===\n\n');

% RMSE values
fprintf('\nRoot Mean Square Error (RMSE):\n');
fprintf('                        Single Sensor | Dual Sensors\n');
fprintf('Position Error (m):      %12.2f | %11.2f\n', rmse_pos_single, rmse_pos_dual);
fprintf('Range Error (m):         %12.2f | %11.2f\n', rmse_r_single, rmse_r_dual);
fprintf('Angle Error (deg):       %12.2f | %11.2f\n', rad2deg(rmse_theta_single), rad2deg(rmse_theta_dual));
fprintf('Radial Vel Error (m/s):  %12.2f | %11.2f\n', rmse_vr_single, rmse_vr_dual);
fprintf('Angular Vel Error (deg/s): %12.2f | %11.2f\n', rad2deg(rmse_vtheta_single), rad2deg(rmse_vtheta_dual));

% Mean Absolute Error values
fprintf('\nMean Absolute Error (MAE):\n');
fprintf('                        Single Sensor | Dual Sensors\n');
fprintf('Position Error (m):      %12.2f | %11.2f\n', mae_pos_single, mae_pos_dual);
fprintf('Range Error (m):         %12.2f | %11.2f\n', mae_r_single, mae_r_dual);
fprintf('Angle Error (deg):       %12.2f | %11.2f\n', rad2deg(mae_theta_single), rad2deg(mae_theta_dual));
fprintf('Radial Vel Error (m/s):  %12.2f | %11.2f\n', mae_vr_single, mae_vr_dual);
fprintf('Angular Vel Error (deg/s): %12.2f | %11.2f\n', rad2deg(mae_vtheta_single), rad2deg(mae_vtheta_dual));

% Final Steady-state error values (last 50 time steps)
ss_error_pos_single = mean(error_pos_single(end-49:end));
ss_error_pos_dual = mean(error_pos_dual(end-49:end));

fprintf('\nSteady-state position error (last 50 time steps):\n');
fprintf('Single Sensor: %.2f m\n', ss_error_pos_single);
fprintf('Dual Sensors: %.2f m\n', ss_error_pos_dual);