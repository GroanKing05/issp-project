function [filtered_signal, mse_history, w_history] = sdaf_filter(corrupted_signal, noise_ref, filter_order, mu, max_iterations)
    % Steepest Descent Adaptive Filter for ECG noise removal
    % Inputs:
    %   corrupted_signal - The noisy ECG signal
    %   noise_ref        - The reference noise signal
    %   filter_order     - Filter order (p), resulting in p+1 coefficients: w(0) to w(p)
    %   mu               - Step size (learning rate)
    %   max_iterations   - Maximum number of iterations
    %
    % Outputs:
    %   filtered_signal  - The filtered ECG signal
    %   mse_history      - History of mean square error (moving average)
    %   w_history        - History of filter weights
    
    if nargin < 5
        max_iterations = 1000; % Default max iterations
    end
    
    if nargin < 4
        mu = 0.01; 
    end

    num_coeffs = filter_order + 1;
    corrupted_signal = corrupted_signal(:);
    noise_ref = noise_ref(:);
    
    signal_length = length(corrupted_signal);
    w = zeros(num_coeffs, 1);
    
    filtered_signal = zeros(size(corrupted_signal));
    mse_history = zeros(max_iterations, 1);
    w_history = zeros(num_coeffs, max_iterations);
    
    % delay line matrix for noise reference signal
    X = zeros(signal_length, num_coeffs);
    for i = 1:num_coeffs
        X(i:end, i) = noise_ref(1:end-i+1);
    end
    
    % window for moving average MSE 
    window_size = 50;  
    
    % buffer for squared error 
    error_buffer = zeros(window_size, 1);
    
    % steepest descent algorithm
    for iter = 1:max_iterations
        % Estimate noise using current weights
        estimated_noise = X * w;
        
        % corrupted signal minus estimated noise
        filtered_signal = corrupted_signal - estimated_noise;
        error = filtered_signal;
        squared_errors = error.^2;
        
        % Compute instantaneous MSE 
        instant_mse = mean(squared_errors);
        
        % update error buffer for moving average 
        error_buffer = [instant_mse; error_buffer(1:end-1)];
        
        % MSE as moving average
        if iter < window_size
            mse_history(iter) = mean(error_buffer(1:iter));
        else
            mse_history(iter) = mean(error_buffer);
        end
        
        w_history(:, iter) = w;
        gradient = -2 * (X' * error) / signal_length;
        w = w - mu * gradient;
    end
    
    % final filter output 
    estimated_noise = X * w;
    filtered_signal = corrupted_signal - estimated_noise;
end