%% Initialization
clc; clear; close all;
load("wdbc_data.mat")

%% Functions
function [mind, maxd, standard_data] = standardizeData(data)
    mind = min(data); % No min/max over the dataset
    maxd = max(data); % in order to see dynamics within the vector
    standard_data = (data-mind)/(maxd-mind);
end

function destdData = deStandardizeData(mind, maxd, data)
    destdData = data.*(maxd' - mind') + mind';
end

function global_best_position = pso_linear_optimization(dataset)
    % Parameters
    n_particles = 50; % Number of particles
    n_dimensions = 30; % Number of dimensions (inputs)
    max_iterations = 1000; % Maximum number of iterations
    w1 = 0.9; % Inertia weight initial
    w2 = 0.4; % Intertia weight final
    c1 = 1.7; % Cognitive coefficient
    c2 = 1.75; % Social coefficient
    condition = 10e-2;

    % Initialize particles
    particles = rand(n_particles, n_dimensions);
    velocities = rand(n_particles, n_dimensions);
    personal_best_positions = particles;
    personal_best_scores = inf(n_particles, 1);

    % Initialize global best
    global_best_position = particles(1, :);
    global_best_score = inf;

    % Main loop
    for iteration = 1:max_iterations
        for i = 1:n_particles
            % Evaluates the current particle using the dataset
            current_score = linear_objective_function(particles(i, :), dataset);

            if current_score < condition
                break;
            end

            % Updates personal best
            if current_score < personal_best_scores(i)
                personal_best_scores(i) = current_score;
                personal_best_positions(i, :) = particles(i, :);

                % Updates global best
                if current_score < global_best_score
                    global_best_score = current_score;
                    global_best_position = particles(i, :);
                end
            end

            % Updates velocity and position
            r1 = rand(1, n_dimensions);
            r2 = rand(1, n_dimensions);
            velocities(i, :) = (w1 - (iteration/max_iterations)*(w1 - w2)) * velocities(i, :) + ...
                               c1 * r1 .* (personal_best_positions(i, :) - particles(i, :)) + ...
                               c2 * r2 .* (global_best_position - particles(i, :));
            particles(i, :) = particles(i, :) + velocities(i, :);
        end
        
        % Display progress
        %fprintf('Iteration %d: Best Score = %f\n', iteration, global_best_score);
    end
    fprintf('Best score: %f\n', global_best_score);
end

function score = linear_objective_function(x, dataset)
    % Evaluates the fitness of a particle based on the dataset
    inputs = dataset(:, 2:end);
    actual_outputs = dataset(:, 1);
    predicted_outputs = classifyData(inputs * x', 2, 4); % Linear combination of inputs
    errors = predicted_outputs - actual_outputs;
    score = mean(errors.^2); % Mean squared error
end

function classifiedData = classifyData(data, b0, b1)
    classifiedData = data;
    classifiedData(classifiedData < (b0+b1)/2) = b0;
    classifiedData(classifiedData >= (b0+b1)/2) = b1;  
end

function [m_perc, b_perc] = quantifyData(output, b0, b1)
    tot_l = length(output);
    m_perc = 100* (sum(output(:) == b0)/tot_l);
    b_perc = 100* (sum(output(:) == b1)/tot_l);
end

%% Neural network

k = 20; 
% Creates a cross-validation partition object
cv = cvpartition(size(data, 1), 'KFold', k);

% Number of neurons in the hidden layer
hiddenLayerSize = round(length(data(1, 2:end)));

% Creates the network
net = feedforwardnet(hiddenLayerSize);


% Set the transfer function for the hidden layer to be linear
net.layers{1}.transferFcn = 'purelin'; % Linear transfer function for the hidden layers

% Set the transfer function for the output layer to be linear
net.layers{2}.transferFcn = 'purelin'; % Linear transfer function for the outpout layer


% Configures the network
net.trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation
net.performFcn = 'mse';    % Mean Squared Error performance function

% Loop variables
performance = zeros(k, 1);
cumul_perf = [];
best_perf = inf;
bestNet = 0;

% Train and evaluate the network
for i = 1:k
    % Get the training and test indices
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    % Spliting the data and labels
    X_train = data(trainIdx, 2:end);
    Y_train = data(trainIdx, 1);
    X_test = data(testIdx, 2:end);
    Y_test = data(testIdx, 1);

    % Trains the network
    [net, tr] = train(net, X_train', Y_train');

    % Evaluates the network
    outputs = net(X_test');
    performance = perform(net, Y_test', outputs);

    cumul_perf = [cumul_perf, performance];
    % Display the performance
    fprintf('Network performance: %.4f\n', performance);

    if (performance < best_perf)
        bestNet = net;
    end
end 

% Calculate and display the average performance
avgPerformance = mean(cumul_perf);
fprintf('Average Performance: %.4f\n', avgPerformance);
%}
%load("best_network.mat");

bestOutput = bestNet(data(:, 2:end)');
bestClassOutput = classifyData(bestOutput, 2, 4);
performance = perform(bestNet, data(:, 1)', bestClassOutput);

fprintf('Reference repartition: %.4f m \n', quantifyData(data(:, 1), 2, 4));
fprintf('Output repartition: %.4f m \n', quantifyData(bestClassOutput, 2, 4));
% Neural network output visualization
%{
figure;
hold on;
plot(data(:, 1));
plot(bestClassOutput); 
grid on;
title("Feedforward linear neural network output");
xlabel("Sample n°");
ylabel("Output");
legend("Classification", "Neural Network");
%}

figure;
C = confusionmat(data(:, 1), bestClassOutput');
confusionchart(C);
%% PSO Computation
% Selecting the dataset
k = 10;
cv = cvpartition(size(data, 1), 'KFold', k);

mse_err = Inf;
pso_best_params = zeros(1, 30);

for j=1:k
    % Determining indexes
    trainIdx = training(cv, j);
    testIdx = test(cv, j);
    
    X_train = data(trainIdx, 2:end);
    Y_train = data(trainIdx, 1);
    X_test = data(testIdx, 2:end);
    Y_test = data(testIdx, 1);

    % Data Training
    pso_opt_params = pso_linear_optimization([Y_train, X_train]);
    
    % Performance test
    pso_test = classifyData(sum(pso_opt_params .* X_test, 2), 2, 4);

    err = Y_test - pso_test;

    if mean(err.^2) < mse_err
        pso_best_params = pso_opt_params;
        mse_err = mean(err.^2);
    end
end

% PSO validation
Y_pso = classifyData(sum(pso_best_params.*data(:, 2:end) , 2), 2, 4);
fprintf('Reference repartition: %.4f m \n', quantifyData(data(:, 1), 2, 4));
fprintf('Output repartition: %.4f m \n', quantifyData(Y_pso, 2, 4));

% Visualisation
figure;
plot(data(:,1)); hold on;
plot(Y_pso);
legend("Classification","PSO");
ylim([1, 5]);
grid on;
title("PSO output");
xlabel("Sample n°");
ylabel("Output");

figure;
C2 = confusionmat(data(:, 1), Y_pso);
confusionchart(C2);

%% Decision boundary 
inputWeights = bestNet.IW{1};
inputBias = bestNet.b{1};
func = str2func(bestNet.layers{1}.transferFcn);


figure;
hold on;
for i=1:30
    H = func(inputWeights(i, :)* data(i, 2:end)' + inputBias);
    plot(data(i, 2:end), H);
end

grid on;


