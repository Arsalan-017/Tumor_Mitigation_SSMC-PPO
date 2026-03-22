function [actorWeights, criticWeights] = updateNetworks_with_SMC(states, actions, advantages, returns, actorWeights, criticWeights, learningRate, clipRatio, gradientClipValue, entropyCoeff)
    for t = 1:length(states)
        state = states(t, :)';
        action = actions(t, :)';
        advantage = advantages(t);
        returnValue = returns(t);
        
        % Sliding surface (example)
        s = state(1) - 0.00; 
        %u_smc = -eta * sign(s);
        u_smc = (0.1/1)*(s/(1+exp(-0.01*s)));
        
        % Update actor weights based on SMC control law
        actorLoss = advantage * u_smc;
        
        % Compute actor gradient
        actorGrad = state * action' * actorLoss';
        actorGrad = max(min(actorGrad, gradientClipValue), -gradientClipValue); % Clip gradients
        
        % Update actor weights
        actorWeights = actorWeights - learningRate * actorGrad;
        
        % Compute critic gradient
        criticLoss = (returnValue - criticWeights' * state)^2;
        criticGrad = 2 * (returnValue - criticWeights' * state) * state;
        criticGrad = reshape(criticGrad, size(criticWeights)); % Ensure dimensions are correct
        criticGrad = max(min(criticGrad, gradientClipValue), -gradientClipValue); % Clip gradients
        
        % Update critic weights
        criticWeights = criticWeights - learningRate * criticGrad;
    end
end
