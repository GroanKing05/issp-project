function [filtered_signal, mse_history, w_history] = rls_filter(primary_signal, reference_signal, filter_order, lambda, delta, max_iterations)
    % RLS_FILTER Recursive Least Squares adaptive filter implementation
    %   [FILTERED_SIGNAL, MSE_HISTORY, W_HISTORY] = RLS_FILTER(PRIMARY_SIGNAL, 
    %   REFERENCE_SIGNAL, FILTER_ORDER, LAMBDA, DELTA, MAX_ITERATIONS)
    %
    %   Inputs:
    %       primary_signal   - The corrupted signal (desired + noise)
    %       reference_signal - The reference noise signal
    %       filter_order     - Filter order (number of filter coefficients + 1)
    %       lambda           - Forgetting factor 
    %       delta            - Initial value for P(0) matrix (regularization parameter)
    %       max_iterations   - Maximum number of iterations
    %
    %   Outputs:
    %       filtered_signal  - The filtered output signal
    %       mse_history      - Mean squared error at each iteration
    %       w_history        - Filter weights at each iteration
    
    w = zeros(filter_order + 1, 1); 
    
    % inverse correlation matrix
    P = (1/delta) * eye(filter_order + 1); % P(0) = δ{^-1}I
    
    signal_length = length(primary_signal);
    filtered_signal = zeros(signal_length, 1);
    mse_history = zeros(min(max_iterations, signal_length), 1);
    w_history = zeros(filter_order + 1, min(max_iterations, signal_length));
    
    % Initialize error sum
    error_sum = 0;
    
    for n = 1:min(max_iterations, signal_length)
        % current input vector 
        if n <= filter_order
            % Pad with zeros for the initial samples
            x = [reference_signal(1:n); zeros(filter_order + 1 - n, 1)];
        else
            x = reference_signal(n:-1:n-filter_order);
        end
        
        % gain vector
        z = P * x; % z(n) = P(n-1)x(n)
        g = z / (lambda + x' * z); % g(n) = z(n) / (λ + x'(n)z(n))
        
        % output of the adaptive filter 
        y_hat = w' * x; % (estimated noise)  
        e = primary_signal(n) - y_hat; % apriori error
        w = w + g * e; 
        
        % Update the inverse correlation matrix
        P = (1/lambda) * (P - g * x' * P); % P(n) = (1/λ)[P(n-1) - g(n)z'(n)]
        
        % error signal (cleaned signal)
        filtered_signal(n) = e;
        
        % cumulative error sum
        error_sum = error_sum + e^2;
        mse_history(n) = error_sum / n;  % Time-averaged MSE
        w_history(:, n) = w;
    end
end