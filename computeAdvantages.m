function [advantages, returns] = computeAdvantages(rewards, values, gamma)
    returns = zeros(size(rewards));
    advantages = zeros(size(rewards));
    G = 0;
    for t = length(rewards):-1:1
        G = rewards(t) + gamma * G;
        returns(t) = G;
        advantages(t) = G - values(t);
    end
end