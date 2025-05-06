function [filtered_signal, mse_history, w_history] = sdaf_filter(corrupted_signal, noise_ref, filter_order, mu, max_iterations)
    % SDAF_FILTER Steepest Descent Adaptive Filter for ECG noise removal
    %
    % Inputs:
    %   corrupted_signal - The noisy ECG signal
    %   noise_ref        - The reference noise signal
    %   filter_order     - Filter order (p), resulting in p+1 coefficients: w(0) to w(p)
    %   mu               - Step size (learning rate)
    %   max_iterations   - Maximum number of iterations
    %
    % Outputs:
    %   filtered_signal  - The filtered ECG signal
    %   mse_history      - History of mean square error
    %   w_history        - History of filter weights
    
    % Input validation
    if nargin < 5
        max_iterations = 1000; % Default max iterations
    end
    
    if nargin < 4
        mu = 0.01; % Default step size
    end
    
    % Filter order p means p+1 coefficients (w0 to wp)
    num_coeffs = filter_order + 1;
    
    % Ensure signals are column vectors
    corrupted_signal = corrupted_signal(:);
    noise_ref = noise_ref(:);
    
    % Get signal length
    signal_length = length(corrupted_signal);
    
    % Initialize filter weights to zeros - now using p+1 coefficients
    w = zeros(num_coeffs, 1);
    
    % Pre-allocate memory for outputs
    filtered_signal = zeros(size(corrupted_signal));
    mse_history = zeros(max_iterations, 1);
    w_history = zeros(num_coeffs, max_iterations);
    
    % Create delay line matrix for noise reference signal
    X = zeros(signal_length, num_coeffs);
    for i = 1:num_coeffs
        X(i:end, i) = noise_ref(1:end-i+1);
    end
    
    % Main steepest descent algorithm
    for iter = 1:max_iterations
        % Estimate noise using current filter weights
        estimated_noise = X * w;
        
        % Filtered signal is the corrupted signal minus estimated noise
        filtered_signal = corrupted_signal - estimated_noise;
        
        % Calculate error (we assume this is close to the original signal)
        error = filtered_signal;
        
        % Compute mean square error
        mse = mean(error.^2);
        mse_history(iter) = mse;
        
        % Store current weights
        w_history(:, iter) = w;
        
        % Check for convergence (optional)
        if iter > 1 && abs(mse_history(iter) - mse_history(iter-1)) < 1e-8
            % Truncate outputs to actual iterations
            mse_history = mse_history(1:iter);
            w_history = w_history(:, 1:iter);
            break;
        end
        
        % Update weights using steepest descent
        gradient = -2 * (X' * error) / signal_length;
        w = w - mu * gradient;
    end
    
    % Final filter output with last weight update
    estimated_noise = X * w;
    filtered_signal = corrupted_signal - estimated_noise;
    
    end