% Define the clip function
function y = clip(x, minVal, maxVal)
    y = max(min(x, maxVal), minVal);
end
