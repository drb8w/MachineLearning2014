function MyPerceptron()
    clc, clear, close all

    n = 100;
    d = 2;
    
    %Mu_1 = [14,14,14,14]; 
    %Sigma_1 = [5,4,3,2];
    Mu_1 = [34,34,34,34]; 
    Sigma_1 = [10,8,6,4];    
    
    Mu_2 = [2, 2, 2, 2]; 
    %Sigma_2 = [2,3,4,5];
    %Sigma_2 = [3,3,3,3];
    Sigma_2 = [4,4,4,4];

    Seed = [2,2,2,2];
    
    [m_Sigma, n_Sigma] = size(Sigma_1);
    
    for index_DataSet=1:n_Sigma

        [data, target] = genData(n, d, Mu_1(index_DataSet), Sigma_1(index_DataSet), Mu_2(index_DataSet), Sigma_2(index_DataSet), Seed(index_DataSet));
        [data_1, data_2] = separateData(data, target);

        no_figure_dataset = index_DataSet*100;
        %displayData(data_1, data_2, no_figure_dataset);
        displayData(data_1, data_2, Mu_1(index_DataSet), Sigma_1(index_DataSet),Mu_2(index_DataSet), Sigma_2(index_DataSet), no_figure_dataset)
        close(no_figure_dataset);

        X = createHomogenData(data);
        t = target;
        maxIts = 10000;

        Gamma = 0.1:0.1:4;
        %Gamma = 0.001:0.001:0.04;
        %Gamma = 1:1;
        [Gamma_m,Gamma_n] = size(Gamma);
        Iterations_online = zeros(Gamma_n,1);
        Iterations_batch = zeros(Gamma_n,1);

        for index_Gamma=1:Gamma_n
            gamma = Gamma(index_Gamma);
            online = true;
            [w_online, its_online] = percTrain(X,t,maxIts, online, gamma);
            online = false;
            [w_batch, its_batch] = percTrain(X,t,maxIts, online, gamma);
            Iterations_online(index_Gamma) = its_online;
            Iterations_batch(index_Gamma) = its_batch;
            no_figure_dataAndBorder = no_figure_dataset+index_Gamma;
            displayDataAndBorder(data_1, data_2, w_online, w_batch, gamma, its_online, its_batch, no_figure_dataAndBorder);
            close(no_figure_dataAndBorder);
        end

        no_figure_GammaToIts = no_figure_dataset+50;
        displayGammaToIterations(Gamma, Iterations_online, Iterations_batch, no_figure_GammaToIts);
        close(no_figure_GammaToIts);

    end
    
end

function [data, target] = genData(n, d, mu_1, sigma_1, mu_2, sigma_2, seed)
    %seed = 2;
    rng(seed);
    data = zeros(n,d);
    target = zeros(1,n);
    n_1 = uint32(n/2);
    for i_n1=1:n_1
        %x_i1 = (randn(1,d)+mu_1)*sigma_1;
        x_i1 = mu_1+sigma_1*randn(1,d);
        data(i_n1,:) = x_i1(:);
        target(1,i_n1)= int8(1);
    end
    
    for i_n2=(n_1+1):n
        %x_i2 = (randn(1,d)+mu_2)*sigma_2;
        x_i2 = mu_2+sigma_2*randn(1,d);
        data(i_n2,:) = x_i2(:);
        target(1,i_n2)= int8(-1);
    end

end

function [data_1, data_2] = separateData(data, target)
    % conditionally separate data depending on target
    
    [m,n] = size(data);
    
    %%HACK
    %data_1 = data(1:m/2,:);
    %data_2 = data((m/2+1):m,:);
    
    data_1 = data((target==1), : );
    data_2 = data((target==-1), : );    
end

function displayData(data_1, data_2, mu_1, Sigma_1, mu_2, Sigma_2, no_figure)

    figure(no_figure);
    
    hold on;
    
    grid
    plot(data_1(:,1),data_1(:,2),'ro', data_2(:,1),data_2(:,2),'gx');
    
    titleStr = strcat('data set; mu_1=', num2str(mu_1),', Sigma_1=', num2str(Sigma_1),'; mu_2=', num2str(mu_2),', Sigma_2=', num2str(Sigma_2));
    title (titleStr);
    
    hold off;
    
    figureName = strcat('DataSet_', int2str(no_figure));
    
    saveas(gcf, figureName,'jpg');

end

function displayDataAndBorder(data_1, data_2, w_online, w_batch, gamma, its_online, its_batch, no_figure)

    figure(no_figure);
    
    hold on;
    
    grid
    
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
    
    titleStr = strcat('data set and decision boundaries with gamma = ',num2str(gamma));
    
    title (titleStr);
    
    %legend('w_{online}(1)+w_{online}(2)*x+w_{online}(3)*y = 0', 'w_{batch}(1)+w_{batch}(2)*x+w_{batch}(3)*y = 0','Location', 'SouthEast');
    str_online = strcat('w_{online}(1)+w_{online}(2)*x+w_{online}(3)*y = 0; iterations: ', int2str(its_online));
    str_batch = strcat('w_{batch}(1)+w_{batch}(2)*x+w_{batch}(3)*y = 0; iterations: ', int2str(its_batch));
    legend(str_online, str_batch, 'Location', 'SouthEast');
    
    plot(data_1(:,1),data_1(:,2),'ro', data_2(:,1),data_2(:,2),'gx');
    
    hold off;
    
    %figureName = strcat('DataSetAndDecisionBoundary_0', int2str(no_figure),'_',num2str(gamma));
    figureName = strcat('DataSetAndDecisionBoundary_', int2str(no_figure));
    
    saveas(gcf, figureName,'jpg');
end


function displayGammaToIterations(Gamma, Iterations_online, Iterations_batch, no_figure)
    figure(no_figure);
    hold on;
    grid
    
    titleStr = strcat('influence of gamma on no of iterations');
    title (titleStr);
    
    plot(Gamma(:),Iterations_online(:),'r', Gamma(:),Iterations_batch(:),'g');

    str_online = strcat('iterations_{online}');
    str_batch = strcat('iterations_{batch}');
    legend(str_online, str_batch, 'Location', 'SouthEast');

    hold off;
    
    figureName = strcat('GammaToIterations_0', int2str(no_figure));
    
    saveas(gcf, figureName,'jpg');

end

function X = createHomogenData(data)
    [m,n] = size(data);
    
    X = [ones(m,1),data];
end

function [w, its] = percTrain(X,t,maxIts, online, gamma)
   if online == true
       [w, its] = onlineLearn(X,t,maxIts, gamma);
   else
       [w, its] = batchLearn(X,t,maxIts, gamma);
   end
end

function [w, its] = onlineLearn(X,t,maxIts, gamma)
    %online
    % 1. Initialize w, gamma
    % 2. do
    % 3. for i = 1 to N
    % 4. if w'*(x_i*t_i ) ? 0 (misclassified ith pattern)
    % 5. w = w + gamma*x_i*t_i
    % 6. end if
    % 7. end for
    % 8. until all patterns correctly classified

    %X = zeros(n,d);
    %t = zeros(1,n);
    [N,d] = size(X);
    %w = ones(d,1); % column vector
    w = zeros(d,1); % column vector

    misclassfied = true;
    its = 0;
    while misclassfied == true && its <= maxIts
        misclassfied = false;
        for i = 1:N
            if w'*(X(i,:)'*t(1,i)) <= 0 % (misclassified ith pattern)
                w = w + gamma*X(i,:)'*t(1,i);
                misclassfied = true;
            end
        end
        its = its + 1;    
    end
end

function [w, its] = batchLearn(X,t,maxIts, gamma)

    % batch
    % 1. Initialize w, gamma
    % 2. do
    % 3. w_delta = 0
    % 4. for i = 1 to N
    % 5. if w'*(x_i*t_i) <= 0 (misclassified ith pattern)
    % 6. w_delta = w_delta + x_i*t_i
    % 7. end if
    % 8. end for
    % 9. w = w + gamma*w_delta
    % 10. until all patterns correctly classified
    
    %X = zeros(n,d);
    %t = zeros(1,n);
    [N,d] = size(X);
    %w = ones(d,1); % column vector;
    w = zeros(d,1); % column vector;

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
        w = w + gamma*w_delta;
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
