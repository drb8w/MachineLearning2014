function MyPerceptron()
    clc, clear, close all

    Mu_1 = [14,14,14,14]; 
    Sigma_1 = [5,4,3,2];

    Mu_2 = [2, 2, 2, 2]; 
    Sigma_2 = [2,3,4,5];

    %Seed = [2,3,4,5];
    Seed = [2,2,2,2];

    for index=1:4
        MySpecialPerceptron(Mu_1(index), Sigma_1(index),Mu_2(index), Sigma_2(index), Seed(index), index);
    end
end


function MySpecialPerceptron(mu_1,sigma_1,mu_2,sigma_2, seed, no_figure)
    n = 100;
    d = 2;
    [data, target] = genData(n, d, mu_1, sigma_1, mu_2, sigma_2, seed);
    [data_1, data_2] = separateData(data, target);
    
    displayData(data_1, data_2, no_figure);
    
    close(no_figure);

    X = createHomogenData(data);
    t = target;
    maxIts = 10000;
    Gamma = 1;
    
    online = true;
    w_online = percTrain(X,t,maxIts, online, Gamma);
    
    online = false;
    w_batch = percTrain(X,t,maxIts, online, Gamma);
    
%     % test perceptron
%     x_20 = X(20,:);
%     t_20_o = perc(w_online,x_20);
%     t_20_b = perc(w_batch,x_20);
%     t_20 = t(20);
%     
%     x_60 = X(60,:);
%     t_60_o = perc(w_online,x_60);
%     t_60_b = perc(w_batch,x_60);
%     t_60 = t(60);
    
    displayDataAndBorder(data_1, data_2, w_online, w_batch, no_figure);
end

function [data, target] = genData(n, d, mu_1, sigma_1, mu_2, sigma_2, seed)
    %seed = 2;
    rng(seed);
    data = zeros(n,d);
    target = zeros(1,n);
    n_1 = uint32(n/2);
    for i_n1=1:n_1
        x_i1 = (randn(1,d)+mu_1)*sigma_1;
        data(i_n1,:) = x_i1(:);
        target(1,i_n1)= int8(1);
    end
    
    for i_n2=(n_1+1):n
        x_i2 = (randn(1,d)+mu_2)*sigma_2;
        data(i_n2,:) = x_i2(:);
        target(1,i_n2)= int8(-1);
    end

end

function [data_1, data_2] = separateData(data, target)
    % conditionally separate data depending on target
    
    [m,n] = size(data);
    
    %HACK
    data_1 = data(1:m/2,:);
    data_2 = data((m/2+1):m,:);
end

function displayData(data_1, data_2, no_figure)

    figure(no_figure);
    
    hold on;
    
    grid
    plot(data_1(:,1),data_1(:,2),'ro', data_2(:,1),data_2(:,2),'gx');
    
    hold off;
    
    figureName = strcat('DataSet_0', int2str(no_figure));
    
    saveas(gcf, figureName,'jpg');

end


function displayDataAndBorder(data_1, data_2, w_online, w_batch, no_figure)

    figure(no_figure);
    
    hold on;
    
    grid
    plot(data_1(:,1),data_1(:,2),'ro', data_2(:,1),data_2(:,2),'gx');
    
    x_min = min(min(data_1(:,1)), min(data_2(:,1)));
    x_max = max(max(data_1(:,1)), max(data_2(:,1)));    
    if ~isnan(w_online)
        p1 = ezplot(@(x,y) w_online(1)+w_online(2)*x+w_online(3)*y,[x_min,x_max]);
        set (p1,'LineWidth',1);
        set (p1, 'Color', 'cyan');
    end
    
    if ~isnan(w_batch)
        p2 = ezplot(@(x,y) w_batch(1)+w_batch(2)*x+w_batch(3)*y,[x_min,x_max]);
        set (p2,'LineWidth',1);
        set (p2, 'Color', 'magenta');
    end
    
    hold off;
    
    figureName = strcat('DataSetAndDecisionBoundary_0', int2str(no_figure));
    
    saveas(gcf, figureName,'jpg');
end

function X = createHomogenData(data)
    [m,n] = size(data);
    
    X = [ones(m,1),data];
end

function w = percTrain(X,t,maxIts, online, Gamma)
   if online == true
       w = onlineLearn(X,t,maxIts, Gamma);
   else
       w = batchLearn(X,t,maxIts, Gamma);
   end
end

function w = onlineLearn(X,t,maxIts, Gamma)
    %online
    % 1. Initialize w, Gamma
    % 2. do
    % 3. for i = 1 to N
    % 4. if w'*(x_i*t_i ) ? 0 (misclassified ith pattern)
    % 5. w = w + Gamma*x_i*t_i
    % 6. end if
    % 7. end for
    % 8. until all patterns correctly classified

    %X = zeros(n,d);
    %t = zeros(1,n);
    [N,d] = size(X);
    w = ones(d,1); % column vector

    misclassfied = true;
    its = 0;
    while misclassfied == true && its <= maxIts
        misclassfied = false;
        for i = 1:N
            if w'*(X(i,:)'*t(1,i)) <= 0 % (misclassified ith pattern)
                w = w + Gamma*X(i,:)'*t(1,i);
                misclassfied = true;
            end
        end
        its = its + 1;    
    end
end

function w = batchLearn(X,t,maxIts, Gamma)

    % batch
    % 1. Initialize w, Gamma
    % 2. do
    % 3. w_delta = 0
    % 4. for i = 1 to N
    % 5. if w'*(x_i*t_i) <= 0 (misclassified ith pattern)
    % 6. w_delta = w_delta + x_i*t_i
    % 7. end if
    % 8. end for
    % 9. w = w + Gamma*w_delta
    % 10. until all patterns correctly classified
    
    %X = zeros(n,d);
    %t = zeros(1,n);
    [N,d] = size(X);
    w = ones(d,1); % column vector;

    misclassfied = true;
    its = 0;
    while misclassfied == true && its <= maxIts
        misclassfied = false;
        w_delta = zeros(d,1); % column vector
        for i = 1:N
            if w'*(X(i,:)'*t(1,i)) <= 0 % (misclassified ith pattern)
                w_delta = w_delta + X(i,:)'*t(1,i);
                misclassfied = true;
            end
        end
        w = w + Gamma*w_delta;
        its = its + 1;    
    end

end

function t = perc(w,x)
    % x is a homogen column vector
    % signum function
    if x*w > 0
        t = 1;
    else
        t = -1;
    end

end
