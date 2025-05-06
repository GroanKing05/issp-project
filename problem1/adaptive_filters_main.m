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

%% Apply final converged filter taps to the noisy inputs and plot comparisons
% Get the final filter coefficients from each algorithm
final_w_sdaf1 = w_history1_sdaf(:, end);
final_w_lms1 = w_history1_lms(:, end);
final_w_rls1 = w_history1_rls(:, end);

final_w_sdaf2 = w_history2_sdaf(:, end);
final_w_lms2 = w_history2_lms(:, end);
final_w_rls2 = w_history2_rls(:, end);

% Apply final converged filters to Signal 1
filtered_final_sdaf1 = zeros(size(y_new_1));
filtered_final_lms1 = zeros(size(y_new_1));
filtered_final_rls1 = zeros(size(y_new_1));

for n = 1:length(y_new_1)
    % Create input vector with appropriate padding for short indices
    if n <= p
        x = [v2(1:n); zeros(p+1-n, 1)];
    else
        x = v2(n:-1:n-p);
    end
    
    % Apply each filter
    y_sdaf = final_w_sdaf1' * x;
    y_lms = final_w_lms1' * x;
    y_rls = final_w_rls1' * x;
    
    % Store filtered outputs
    filtered_final_sdaf1(n) = y_new_1(n) - y_sdaf;
    filtered_final_lms1(n) = y_new_1(n) - y_lms;
    filtered_final_rls1(n) = y_new_1(n) - y_rls;
end

% Apply final converged filters to Signal 2
filtered_final_sdaf2 = zeros(size(y_new_2));
filtered_final_lms2 = zeros(size(y_new_2));
filtered_final_rls2 = zeros(size(y_new_2));

for n = 1:length(y_new_2)
    % Create input vector with appropriate padding for short indices
    if n <= p
        x = [w2(1:n); zeros(p+1-n, 1)];
    else
        x = w2(n:-1:n-p);
    end
    
    % Apply each filter
    y_sdaf = final_w_sdaf2' * x;
    y_lms = final_w_lms2' * x;
    y_rls = final_w_rls2' * x;
    
    % Store filtered outputs
    filtered_final_sdaf2(n) = y_new_2(n) - y_sdaf;
    filtered_final_lms2(n) = y_new_2(n) - y_lms;
    filtered_final_rls2(n) = y_new_2(n) - y_rls;
end

% Plot comparison of filtered outputs for Signal 1
figure;
subplot(4, 1, 1);
plot(y_new_1);
title('Corrupted ECG Signal 1');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_1)]);

subplot(4, 1, 2);
plot(filtered_final_sdaf1);
title('Filtered with SDAF Final Coefficients');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_final_sdaf1)]);

subplot(4, 1, 3);
plot(filtered_final_lms1);
title('Filtered with LMS Final Coefficients');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_final_lms1)]);

subplot(4, 1, 4);
plot(filtered_final_rls1);
title('Filtered with RLS Final Coefficients');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_final_rls1)]);

% Plot comparison of filtered outputs for Signal 2
figure;
subplot(4, 1, 1);
plot(y_new_2);
title('Corrupted ECG Signal 2');
ylabel('Amplitude');
grid on;
xlim([0 length(y_new_2)]);

subplot(4, 1, 2);
plot(filtered_final_sdaf2);
title('Filtered with SDAF Final Coefficients');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_final_sdaf2)]);

subplot(4, 1, 3);
plot(filtered_final_lms2);
title('Filtered with LMS Final Coefficients');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_final_lms2)]);

subplot(4, 1, 4);
plot(filtered_final_rls2);
title('Filtered with RLS Final Coefficients');
ylabel('Amplitude');
grid on;
xlim([0 length(filtered_final_rls2)]);

% Calculate MSE for each final filter
mse_final_sdaf1 = mean((filtered_final_sdaf1).^2);
mse_final_lms1 = mean((filtered_final_lms1).^2);
mse_final_rls1 = mean((filtered_final_rls1).^2);

mse_final_sdaf2 = mean((filtered_final_sdaf2).^2);
mse_final_lms2 = mean((filtered_final_lms2).^2);
mse_final_rls2 = mean((filtered_final_rls2).^2);

% Display final filter MSE values
fprintf('\nFinal Filter Application MSE for Signal 1:\n');
fprintf('SDAF: %.6f\n', mse_final_sdaf1);
fprintf('LMS: %.6f\n', mse_final_lms1);
fprintf('RLS: %.6f\n', mse_final_rls1);

fprintf('\nFinal Filter Application MSE for Signal 2:\n');
fprintf('SDAF: %.6f\n', mse_final_sdaf2);
fprintf('LMS: %.6f\n', mse_final_lms2);
fprintf('RLS: %.6f\n', mse_final_rls2);

% Print the final filter coefficients
fprintf('\nFinal Filter Coefficients for Signal 1:\n');
fprintf('SDAF: '); fprintf('%.6f ', final_w_sdaf1); fprintf('\n');
fprintf('LMS: '); fprintf('%.6f ', final_w_lms1); fprintf('\n');
fprintf('RLS: '); fprintf('%.6f ', final_w_rls1); fprintf('\n');

fprintf('\nFinal Filter Coefficients for Signal 2:\n');
fprintf('SDAF: '); fprintf('%.6f ', final_w_sdaf2); fprintf('\n');
fprintf('LMS: '); fprintf('%.6f ', final_w_lms2); fprintf('\n');
fprintf('RLS: '); fprintf('%.6f ', final_w_rls2); fprintf('\n');

% %% Plot windowed power of filtered outputs on a single plot
% % Set window size for power analysis
% window_size = 85;  % Based on signal characteristics (30 peaks in 2560 samples)
% 
% % Function to calculate moving average
% moving_avg = @(x, win_size) filter(ones(1, win_size)/win_size, 1, x);
% 
% % Calculate windowed power for Signal 1
% windowed_power_sdaf1 = moving_avg(filtered_final_sdaf1.^2, window_size);
% windowed_power_lms1 = moving_avg(filtered_final_lms1.^2, window_size);
% windowed_power_rls1 = moving_avg(filtered_final_rls1.^2, window_size);
% 
% % Calculate windowed power for Signal 2
% windowed_power_sdaf2 = moving_avg(filtered_final_sdaf2.^2, window_size);
% windowed_power_lms2 = moving_avg(filtered_final_lms2.^2, window_size);
% windowed_power_rls2 = moving_avg(filtered_final_rls2.^2, window_size);
% 
% % Account for filter delay
% delay = floor(window_size/2);
% 
% % Plot overlaid windowed power for Signal 1
% figure;
% plot(windowed_power_sdaf1, 'b');
% hold on;
% plot(windowed_power_lms1, 'r');
% plot(windowed_power_rls1, 'g');
% title('Windowed Power Comparison - Signal 1');
% xlabel('Sample');
% ylabel('Power');
% legend('SDAF', 'LMS', 'RLS');
% grid on;
% xlim([window_size length(windowed_power_sdaf1)]);
% 
% % Plot overlaid windowed power for Signal 2
% figure;
% plot(windowed_power_sdaf2, 'b');
% hold on;
% plot(windowed_power_lms2, 'r');
% plot(windowed_power_rls2, 'g');
% title('Windowed Power Comparison - Signal 2');
% xlabel('Sample');
% ylabel('Power');
% legend('SDAF', 'LMS', 'RLS');
% grid on;
% xlim([window_size length(windowed_power_sdaf2)]);