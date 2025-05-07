function [filtered_signal, mse_history, w_history] = lms_filter(primary_signal, reference_signal, filter_order, mu, max_iterations)
    % LMS_FILTER Least Mean Squares adaptive filter implementation
    %   [FILTERED_SIGNAL, MSE_HISTORY, W_HISTORY] = LMS_FILTER(PRIMARY_SIGNAL, 
    %   REFERENCE_SIGNAL, FILTER_ORDER, MU, MAX_ITERATIONS)
    %
    %   Inputs:
    %       primary_signal   - The corrupted signal (desired + noise)
    %       reference_signal - The reference noise signal
    %       filter_order     - Filter order (number of filter coefficients - 1)
    %       mu               - Step size (learning rate)
    %       max_iterations   - Maximum number of iterations
    %
    %   Outputs:
    %       filtered_signal  - The filtered output signal
    %       mse_history      - Mean squared error at each iteration
    %       w_history        - Filter weights at each iteration
    
    % Initialize filter coefficients
    w = zeros(filter_order + 1, 1);
    
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
        
        % Calculate filter output - noise estimate
        y_hat = w' * x;
        
        % Calculate error (desired signal estimate)
        e = primary_signal(n) - y_hat;
        
        % Store the error signal (which is our cleaned signal)
        filtered_signal(n) = e;
        
        % Update filter coefficients using LMS algorithm
        w = w + mu * e * x;
        
        % Update cumulative error sum and calculate MSE
        error_sum = error_sum + e^2;
        mse_history(n) = error_sum / n;  % Time-averaged MSE
        
        % Store filter weights
        w_history(:, n) = w;
    end
    
    % % If we haven't processed the entire signal, use the final weights to process the rest
    % if max_iterations < signal_length
    %     for n = (max_iterations+1):signal_length
    %         if n <= filter_order
    %             x = [reference_signal(1:n); zeros(filter_order + 1 - n, 1)];
    %         else
    %             x = reference_signal(n:-1:n-filter_order);
    %         end
            
    %         y_hat = w' * x;
    %         e = primary_signal(n) - y_hat;
    %         filtered_signal(n) = e;
    %     end
    % end
    % end