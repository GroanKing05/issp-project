% Kalman Filter Implementation for tracking Siva's car
close all; clear all; clc;
load('range1.mat');
load('angle1.mat');
angle1_rad = deg2rad(angle1);
dt = 1;

% Initial state and covariance
x = [500; 0; 0; 0]; % [r; theta; vr; vtheta]
P = diag([1000, 10, 50, 10]);
Q_w = zeros(4, 4);

% Measurement noise covariance
angle_var_rad = (deg2rad(4))^2; 
Q_v = diag([2500, angle_var_rad]);

% State transition matrix
A = [1, 0, dt, 0;
     0, 1, 0, dt;
     0, 0, 1, 0;
     0, 0, 0, 1];

% Observation matrix
C = [1, 0, 0, 0;
     0, 1, 0, 0];

% Storage for results
x_history = zeros(4, length(range1));
x_history(:, 1) = x;

% Kalman filter loop
for n = 2:length(range1)
    % prediction 
    x_pred = A * x;
    P_pred = A * P * A' + Q_w;
    
    % update
    y = [range1(n); angle1_rad(n)];
    y_diff = y - C * x_pred;
    S = C * P_pred * C' + Q_v;
    K = P_pred * C' * inv(S);
    
    x = x_pred + K * y_diff;
    P = (eye(4) - K * C) * P_pred;
    
    % store results
    x_history(:, n) = x;
end

% Convert to Cartesian coordinates for plots
x_cart = x_history(1,:) .* cos(x_history(2,:));
y_cart = x_history(1,:) .* sin(x_history(2,:));

% Plot estimated trajectory
figure;
plot(x_cart, y_cart);
hold on;
plot(0, 0, 'ko', 'MarkerFaceColor', 'k');  % Origin (Madhuri's position)
xlabel('X position (m)');
ylabel('Y position (m)');
title('Estimated Trajectory of Siva''s Car');
legend('Estimated Trajectory', 'Origin (Madhuri''s Position)');
axis equal;
grid on;

% Plot estimated velocities
figure;
subplot(2,1,1);
plot(0:length(range1)-1, x_history(3,:));
ylabel('Radial velocity (m/s)');
title('Estimated Radial Velocity');
grid on;

subplot(2,1,2);
plot(0:length(range1)-1, rad2deg(x_history(4,:))); % Convert to deg/s 
ylabel('Angular velocity (deg/s)');
xlabel('Time (s)');
title('Estimated Angular Velocity');
grid on;

% effect of changing initial covariance
cov_scales = [0.1, 1, 10];
colors = {'r', 'g', 'b'};
labels = cell(length(cov_scales), 1);

figure;
hold on;
for i = 1:length(cov_scales)
    scale = cov_scales(i);
    P_init = diag([1000, 10, 50, 10]) * scale;

    % Reset state and initialize covariance with new scale
    x_alt = [500; 0; 0; 0];
    P_alt = P_init;

    % store alternative state history
    x_alt_history = zeros(4, length(range1));
    x_alt_history(:, 1) = x_alt;

    % Run Kalman filter with new covariance
    for n = 2:length(range1)
        % prediction
        x_pred = A * x_alt;
        P_pred = A * P_alt * A' + Q_w;

        % update
        y = [range1(n); angle1_rad(n)];
        y_diff = y - C * x_pred;

        S = C * P_pred * C' + Q_v;
        K = P_pred * C' / S;

        x_alt = x_pred + K * y_diff;
        P_alt = (eye(4) - K * C) * P_pred;

        % store results
        x_alt_history(:, n) = x_alt;
    end

    % Cartesian coordinates
    x_alt_cart = x_alt_history(1,:) .* cos(x_alt_history(2,:));
    y_alt_cart = x_alt_history(1,:) .* sin(x_alt_history(2,:));

    % Plot trajectory with different covariance scale
    plot(x_alt_cart, y_alt_cart, colors{i});
    labels{i} = ['Scale = ', num2str(scale)];
end

plot(0, 0, 'ko', 'MarkerFaceColor', 'k');  % Origin
grid on;
xlabel('X position (m)');
ylabel('Y position (m)');
title('Effect of Initial Covariance on Estimated Trajectory');
legend([labels; {'Origin'}]);
axis equal;

% Plot velocity convergence with different covariances
figure;
subplot(2,1,1);
hold on;
for i = 1:length(cov_scales)
    scale = cov_scales(i);
    P_init = diag([1000, 10, 50, 10]) * scale;

    % Reset state and initialize covariance
    x_alt = [500; 0; 0; 0];
    P_alt = P_init;

    % store alternative results
    x_alt_history = zeros(4, length(range1));
    x_alt_history(:, 1) = x_alt;

    % Run Kalman filter with new covariance
    for n = 2:length(range1)
        % prediction
        x_pred = A * x_alt;
        P_pred = A * P_alt * A' + Q_w;

        % update
        y = [range1(n); angle1_rad(n)];
        y_diff = y - C * x_pred;
        S = C * P_pred * C' + Q_v;
        K = P_pred * C' / S;

        x_alt = x_pred + K * y_diff;
        P_alt = (eye(4) - K * C) * P_pred;

        % store results
        x_alt_history(:, n) = x_alt;
    end

    plot(0:length(range1)-1, x_alt_history(3,:), colors{i});
end
grid on;
xlabel('Time (seconds)');
ylabel('Radial Velocity (m/s)');
title('Effect of Initial Covariance on Radial Velocity Estimate');
legend(labels);

% Run another time to collect weights
% for Kalman gain
K_vr_range_history = zeros(length(cov_scales), length(range1));

for i = 1:length(cov_scales)
    scale = cov_scales(i);
    P_init = diag([1000, 10, 50, 10]) * scale;

    % Reset state and initialize covariance with new scale
    x_alt = [500; 0; 0; 0];
    P_alt = P_init;

    % Run Kalman filter with new covariance and collect coefficients
    for n = 2:length(range1)
        % Prediction step
        x_pred = A * x_alt;
        P_pred = A * P_alt * A' + Q_w;

        % Measurement update
        y = [range1(n); angle1_rad(n)];
        y_diff = y - C * x_pred;

        S = C * P_pred * C' + Q_v;
        K = P_pred * C' / S;
        IKC = eye(4) - K * C;

        % Store K and I-KC values for radial velocity
        K_vr_range_history(i,n) = K(3,1);  % Effect of range measurement on vr
        
        x_alt = x_pred + K * y_diff;
        P_alt = IKC * P_pred;
    end
end

% Plot Kalman gain for radial velocity
subplot(2,1,2);
hold on;
for i = 1:length(cov_scales)
    plot(0:length(range1)-1, K_vr_range_history(i,:), colors{i});
end
grid on;
xlabel('Time (seconds)');
ylabel('K_{v_r,r}');
title('Kalman Gain: Range Measurement → Radial Velocity State');
legend(labels);