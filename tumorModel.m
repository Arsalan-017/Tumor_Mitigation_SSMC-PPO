% Define the tumor model as a function
function [nextState, reward] = tumorModel(state, action, qt, constants)
    % Unpack constants
    r1 = constants.r1; r2 = constants.r2; r3 = constants.r3;
    k1 = constants.k1; k2 = constants.k2; k3 = constants.k3;
    a12 = constants.a12; a13 = constants.a13; a21 = constants.a21; a31 = constants.a31;
    Nc = constants.Nc; Ng = constants.Ng; Nw = constants.Nw;
    d3 = constants.d3; gamma = constants.gamma; sigma = constants.sigma; epsilon = constants.epsilon;
    beta1 = constants.beta1; beta2 = constants.beta2; beta3 = constants.beta3; beta4 = constants.beta4;
    
    % Unpack state
    T = state(1); N = state(2); I = state(3); U = state(4); P = state(5); H = state(6);
    
    % Unpack action and apply constraints
    alpha = clip(action(1), 0, 1); % Limit alpha to max 1
    q = clip(action(2), 0, 1);     % Limit q to max 1
    
    % Compute next state with non-negativity constraints
    T_next = max(T + qt * (r1 * T * (1 - T / k1) - a12 * N * T - a13 * T * I - Nc * (1 - exp(-P)) * T - U * T), 0);
    N_next = max(N + qt * (r2 * N * (1 - N / k2) - a21 * N * T - Ng * (1 - exp(-P)) * N - epsilon * U * N), 0);
    W_next = max(I + qt * (r3 * I * T / (T + k3) - a31 * I * T - d3 * I - Nw * (1 - exp(-P)) * I), 0);
    U_next = U + qt*(-gamma*U + alpha);
    P_next = P + qt*(-sigma*P + q);
    H_next = H + qt*(-beta1 * P - beta2 * U - beta3 * T + beta4 * N);
    
    nextState = [T_next, N_next, W_next, U_next, P_next, H_next];
    
   reward = -(T_next)^2 + 0.1 * (N_next)^2 - 0.05*(W_next)^2 - 0.01 * (alpha^2 + q^2);  % Reward shaping and action smoothing
end
