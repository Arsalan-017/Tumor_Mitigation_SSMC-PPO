function action = policy(state, weights, noise)
    C = state(1); % Tumor cell count
    w_a = 0.0175;
    w_smc = 0.02;
    w_n = 0.015;
    action = weights' * state';
    action = action'.*w_a + [(0.1/1)*(C/(1+exp(-0.5*C))) (10/1)*(C/(1+exp(-0.5*C)))].*w_smc + noise*randn(size(action))'.*w_n; % Add Gaussian noise for exploration
    action = clip(action, 0, 1); % Ensure actions are within valid range
end