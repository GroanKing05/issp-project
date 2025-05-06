% Main script to test adaptive filters on the ECG signals
clc; clear; close all;

% Load the data
load('Signal1.mat');  % Contains y_new_1
load('Signal2.mat');  % Contains y_new_2
load('Noise1.mat');   % Contains v2
load('Noise2.mat');   % Contains w2

% Parameters for the adaptive filters
p = 4;                % Filter order as specified in the problem
mu_sdaf = 0.01;       % Step size for SDAF - adjust as needed
mu_lms = 0.01;        % Step size for LMS - adjust as needed
lambda_rls = 0.99;    % Forgetting factor for RLS (typically 0.95 to 0.99)
delta_rls = 100;      % Regularization parameter for RLS
max_iterations = 3e3; % Maximum iterations

%% SDAF Implementation
% Apply SDAF to Signal1 using Noise1 as reference
[filtered_signal1_sdaf, mse_history1_sdaf, w_history1_sdaf] = sdaf_filter(y_new_1, v2, p, mu_sdaf, max_iterations);

% Apply SDAF to Signal2 using Noise2 as reference
[filtered_signal2_sdaf, mse_history2_sdaf, w_history2_sdaf] = sdaf_filter(y_new_2, w2, p, mu_sdaf, max_iterations);

%% LMS Implementation
% Apply LMS to Signal1 using Noise1 as reference
[filtered_signal1_lms, mse_history1_lms, w_history1_lms] = lms_filter(y_new_1, v2, p, mu_lms, max_iterations);

% Apply LMS to Signal2 using Noise2 as reference
[filtered_signal2_lms, mse_history2_lms, w_history2_lms] = lms_filter(y_new_2, w2, p, mu_lms, max_iterations);

%% RLS Implementation
% Apply RLS to Signal1 using Noise1 as reference
[filtered_signal1_rls, mse_history1_rls, w_history1_rls] = rls_filter(y_new_1, v2, p, lambda_rls, delta_rls, max_iterations);

% Apply RLS to Signal2 using Noise2 as reference
[filtered_signal2_rls, mse_history2_rls, w_history2_rls] = rls_filter(y_new_2, w2, p, lambda_rls, delta_rls, max_iterations);

%% Plot results for Signal1
% SDAF Results
figure;
subplot(3, 1, 1);
plot(y_new_1);
title('Corrupted ECG Signal 1');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_1)]);

subplot(3, 1, 2);
plot(filtered_signal1_sdaf);
title('Filtered ECG Signal 1 (SDAF)');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_signal1_sdaf)]);

subplot(3, 1, 3);
plot(mse_history1_sdaf);
title('SDAF Mean Square Error History');
xlabel('Iteration');
ylabel('MSE');
grid on;
xlim([0 length(mse_history1_sdaf)]);

% LMS Results
figure;
subplot(3, 1, 1);
plot(y_new_1);
title('Corrupted ECG Signal 1');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_1)]);

subplot(3, 1, 2);
plot(filtered_signal1_lms);
title('Filtered ECG Signal 1 (LMS)');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_signal1_lms)]);

subplot(3, 1, 3);
plot(mse_history1_lms);
title('LMS Mean Square Error History');
xlabel('Iteration');
ylabel('MSE');
grid on;
xlim([0 length(mse_history1_lms)]);

% RLS Results
figure;
subplot(3, 1, 1);
plot(y_new_1);
title('Corrupted ECG Signal 1');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_1)]);

subplot(3, 1, 2);
plot(filtered_signal1_rls);
title('Filtered ECG Signal 1 (RLS)');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_signal1_rls)]);

subplot(3, 1, 3);
plot(mse_history1_rls);
title('RLS Mean Square Error History');
xlabel('Iteration');
ylabel('MSE');
grid on;
xlim([0 length(mse_history1_rls)]);

%% Plot results for Signal2
% SDAF Results
figure;
subplot(3, 1, 1);
plot(y_new_2);
title('Corrupted ECG Signal 2');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_2)]);

subplot(3, 1, 2);
plot(filtered_signal2_sdaf);
title('Filtered ECG Signal 2 (SDAF)');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_signal2_sdaf)]);

subplot(3, 1, 3);
plot(mse_history2_sdaf);
title('SDAF Mean Square Error History');
xlabel('Iteration');
ylabel('MSE');
grid on;
xlim([0 length(mse_history2_sdaf)]);

% LMS Results
figure;
subplot(3, 1, 1);
plot(y_new_2);
title('Corrupted ECG Signal 2');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_2)]);

subplot(3, 1, 2);
plot(filtered_signal2_lms);
title('Filtered ECG Signal 2 (LMS)');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_signal2_lms)]);

subplot(3, 1, 3);
plot(mse_history2_lms);
title('LMS Mean Square Error History');
xlabel('Iteration');
ylabel('MSE');
grid on;
xlim([0 length(mse_history2_lms)]);

% RLS Results
figure;
subplot(3, 1, 1);
plot(y_new_2);
title('Corrupted ECG Signal 2');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_2)]);

subplot(3, 1, 2);
plot(filtered_signal2_rls);
title('Filtered ECG Signal 2 (RLS)');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_signal2_rls)]);

subplot(3, 1, 3);
plot(mse_history2_rls);
title('RLS Mean Square Error History');
xlabel('Iteration');
ylabel('MSE');
grid on;
xlim([0 length(mse_history2_rls)]);

%% Plot filter coefficients convergence for Signal1 (SDAF)
figure;
num_coeffs = p + 1; % 5 coefficients (w(0) to w(4))
for i = 1:num_coeffs
    subplot(num_coeffs, 1, i);
    plot(w_history1_sdaf(i, :));
    title(['SDAF Filter Weight w(' num2str(i-1) ') Evolution - Signal 1']);
    xlabel('Iteration');
    ylabel(['w(' num2str(i-1) ')']);
    grid on;
    xlim([0 size(w_history1_sdaf, 2)]);
end

%% Plot filter coefficients convergence for Signal2 (SDAF)
figure;
for i = 1:num_coeffs
    subplot(num_coeffs, 1, i);
    plot(w_history2_sdaf(i, :));
    title(['SDAF Filter Weight w(' num2str(i-1) ') Evolution - Signal 2']);
    xlabel('Iteration');
    ylabel(['w(' num2str(i-1) ')']);
    grid on;
    xlim([0 size(w_history2_sdaf, 2)]);
end

%% Plot filter coefficients convergence for Signal1 (LMS)
figure;
for i = 1:num_coeffs
    subplot(num_coeffs, 1, i);
    plot(w_history1_lms(i, :));
    title(['LMS Filter Weight w(' num2str(i-1) ') Evolution - Signal 1']);
    xlabel('Iteration');
    ylabel(['w(' num2str(i-1) ')']);
    grid on;
    xlim([0 size(w_history1_lms, 2)]);
end

%% Plot filter coefficients convergence for Signal2 (LMS)
figure;
for i = 1:num_coeffs
    subplot(num_coeffs, 1, i);
    plot(w_history2_lms(i, :));
    title(['LMS Filter Weight w(' num2str(i-1) ') Evolution - Signal 2']);
    xlabel('Iteration');
    ylabel(['w(' num2str(i-1) ')']);
    grid on;
    xlim([0 size(w_history2_lms, 2)]);
end

%% Plot filter coefficients convergence for Signal1 (RLS)
figure;
for i = 1:num_coeffs
    subplot(num_coeffs, 1, i);
    plot(w_history1_rls(i, :));
    title(['RLS Filter Weight w(' num2str(i-1) ') Evolution - Signal 1']);
    xlabel('Iteration');
    ylabel(['w(' num2str(i-1) ')']);
    grid on;
    xlim([0 size(w_history1_rls, 2)]);
end

%% Plot filter coefficients convergence for Signal2 (RLS)
figure;
for i = 1:num_coeffs
    subplot(num_coeffs, 1, i);
    plot(w_history2_rls(i, :));
    title(['RLS Filter Weight w(' num2str(i-1) ') Evolution - Signal 2']);
    xlabel('Iteration');
    ylabel(['w(' num2str(i-1) ')']);
    grid on;
    xlim([0 size(w_history2_rls, 2)]);
end

%% Calculate and report final MSE values
% SDAF Statistics
final_mse1_sdaf = mse_history1_sdaf(end);
iterations_to_converge1_sdaf = length(mse_history1_sdaf);
final_mse2_sdaf = mse_history2_sdaf(end);
iterations_to_converge2_sdaf = length(mse_history2_sdaf);

% LMS Statistics
final_mse1_lms = mse_history1_lms(end);
final_mse2_lms = mse_history2_lms(end);

% RLS Statistics
final_mse1_rls = mse_history1_rls(end);
final_mse2_rls = mse_history2_rls(end);

fprintf('Signal 1 Final MSE Values:\n');
fprintf('SDAF: %.6f\n', final_mse1_sdaf);
fprintf('LMS: %.6f\n', final_mse1_lms);
fprintf('RLS: %.6f\n', final_mse1_rls);

fprintf('\nSignal 2 Final MSE Values:\n');
fprintf('SDAF: %.6f\n', final_mse2_sdaf);
fprintf('LMS: %.6f\n', final_mse2_lms);
fprintf('RLS: %.6f\n', final_mse2_rls);