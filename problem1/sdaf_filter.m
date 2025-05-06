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
    %   mse_history      - History of mean square error (moving average)
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
    
    % Set window size for moving average MSE calculation
    window_size = 50;  % Adjust as needed
    
    % Initialize buffer for squared error values
    error_buffer = zeros(window_size, 1);
    
    % Main steepest descent algorithm
    for iter = 1:max_iterations
        % Estimate noise using current filter weights
        estimated_noise = X * w;
        
        % Filtered signal is the corrupted signal minus estimated noise
        filtered_signal = corrupted_signal - estimated_noise;
        
        % Calculate error (we assume this is close to the original signal)
        error = filtered_signal;
        
        % Calculate squared error for each sample
        squared_errors = error.^2;
        
        % Compute instantaneous MSE (mean of all samples)
        instant_mse = mean(squared_errors);
        
        % Update the error buffer for moving average (shift values and add newest)
        error_buffer = [instant_mse; error_buffer(1:end-1)];
        
        % Calculate MSE as moving average
        if iter < window_size
            % For initial iterations before window is filled
            mse_history(iter) = mean(error_buffer(1:iter));
        else
            % Full window moving average
            mse_history(iter) = mean(error_buffer);
        end
        
        % Store current weights
        w_history(:, iter) = w;
        
        % Update weights using steepest descent
        gradient = -2 * (X' * error) / signal_length;
        w = w - mu * gradient;
    end
    
    % Final filter output with last weight update
    estimated_noise = X * w;
    filtered_signal = corrupted_signal - estimated_noise;
    
end