function [filtered_signal, mse_history, w_history] = rls_filter(primary_signal, reference_signal, filter_order, lambda, delta, max_iterations)
    % RLS_FILTER Recursive Least Squares adaptive filter implementation
    %   [FILTERED_SIGNAL, MSE_HISTORY, W_HISTORY] = RLS_FILTER(PRIMARY_SIGNAL, 
    %   REFERENCE_SIGNAL, FILTER_ORDER, LAMBDA, DELTA, MAX_ITERATIONS)
    %
    %   Inputs:
    %       primary_signal   - The corrupted signal (desired + noise)
    %       reference_signal - The reference noise signal
    %       filter_order     - Filter order (number of filter coefficients - 1)
    %       lambda           - Forgetting factor (typically 0.95 to 0.99)
    %       delta            - Initial value for P(0) matrix (regularization parameter)
    %       max_iterations   - Maximum number of iterations
    %
    %   Outputs:
    %       filtered_signal  - The filtered output signal
    %       mse_history      - Mean squared error at each iteration
    %       w_history        - Filter weights at each iteration
    
    % Initialize filter coefficients
    w = zeros(filter_order + 1, 1); % w(0) = 0
    
    % Initialize inverse correlation matrix
    P = (1/delta) * eye(filter_order + 1); % P(0) = δ⁻¹I
    
    % Initialize output arrays
    signal_length = length(primary_signal);
    filtered_signal = zeros(signal_length, 1);
    mse_history = zeros(min(max_iterations, signal_length), 1);
    w_history = zeros(filter_order + 1, min(max_iterations, signal_length));
    
    % Initialize error sum for MSE calculation
    error_sum = 0;
    
    % Process the signal
    for n = 1:min(max_iterations, signal_length)
        % Get current input vector (reference signal)
        if n <= filter_order
            % Pad with zeros for the initial samples
            x = [reference_signal(1:n); zeros(filter_order + 1 - n, 1)];
        else
            x = reference_signal(n:-1:n-filter_order);
        end
        
        % 1. Compute the gain vector
        z = P * x; % z(n) = P(n-1)x(n)
        g = z / (lambda + x' * z); % g(n) = z(n) / (λ + x'(n)z(n))
        
        % 2. Compute the a priori error
        y_hat = w' * x; % filter output
        e = primary_signal(n) - y_hat; % a priori error: α(n) = d(n) - w'(n-1)x(n)
        
        % 3. Update the filter coefficients
        w = w + e * g; % w(n) = w(n-1) + α(n)g(n)
        
        % 4. Update the inverse correlation matrix
        P = (1/lambda) * (P - g * x' * P); % P(n) = (1/λ)[P(n-1) - g(n)z'(n)]
        
        % Store the error signal (which is our cleaned signal)
        filtered_signal(n) = e;
        
        % Update cumulative error sum and calculate MSE
        error_sum = error_sum + e^2;
        mse_history(n) = error_sum / n;  % Time-averaged MSE
        
        % Store filter weights
        w_history(:, n) = w;
    end
    
    % If we haven't processed the entire signal, use the final weights to process the rest
    if max_iterations < signal_length
        for n = (max_iterations+1):signal_length
            if n <= filter_order
                x = [reference_signal(1:n); zeros(filter_order + 1 - n, 1)];
            else
                x = reference_signal(n:-1:n-filter_order);
            end
            
            y_hat = w' * x;
            e = primary_signal(n) - y_hat;
            filtered_signal(n) = e;
        end
    end
    end