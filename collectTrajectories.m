function [states, actions, rewards, nextStates] = collectTrajectories(env, policy, numSteps)
    states = zeros(numSteps, 6);  % Initialize state array
    actions = zeros(numSteps, 2); % Initialize action array
    rewards = zeros(numSteps, 1); % Initialize reward array
    nextStates = zeros(numSteps, 6); % Initialize next state array
    state = resetEnv(); % Reset environment
    for t = 1:numSteps
        action = policy(state);  % Get action from policy
        [nextState, reward] = tumorModel(state, action, 1, env.constants);  % Get next state and reward
        states(t, :) = state;  % Store state
        actions(t, :) = action; % Store action
        rewards(t) = reward;  % Store reward
        nextStates(t, :) = nextState;  % Store next state
        state = nextState;  % Update state
    end
end
