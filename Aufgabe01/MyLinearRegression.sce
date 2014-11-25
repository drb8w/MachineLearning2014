// Lineare Regression

function MyLinearRegression()
    // preinitialisation and setup
    clc, clear, close all
    x_start = 0;
    x_end = 5;
    x_interval = 0.1;
    G = 10;
    [x,y]=generateXY(x_start,x_end,x_interval,G);

    mu = 0;
    Sigma = 0.7;
    [x_t,t]=generateTrainingsSet(x,y,G,mu,Sigma);
    y_t = t';
    dimension_start = 3;
    dimension_end = 9;
    
    // 1.2.2.I - determine w_online_3    
    lambda = 0.001;
    A_3 = createA(x_t,3);
    E_threshold = 0.01;
    maxIts = 200;
    
    [w_online_3, its_online_3]= onlineLMS(A_3, y_t ,lambda, E_threshold, maxIts);
    
    // y_online_3 determined via onlineLMS
    y_online_3 = createPolynomValues(x,w_online_3);    
    
    // 1.2.2.II - determine w^* of quadric error function    
    w_star_3 = compute_w_star(A_3,y_t,lambda);
    
    // y_star_3 determined via pseudoinverse
    y_star_3 = createPolynomValues(x,w_star_3);
    
    // plot y, t, y_online_3 and y_star_3
    no_figureLinReg = 50;
    plotLinearRegressionResults(x, y, t, y_online_3, y_star_3, 3, no_figureLinReg);
    
    // 1.2.2.III - test influence of lambda on convergence of online LMS
    Lambdas = 0.001:0.001:10;
    [m_l,n_l]=size(Lambdas);
    
    //treshold_E_ratio = 1.01;
    Its_online_3 = zeros(1,n_l);
    W_online_3 = zeros(4,n_l);
    for index_lambda=1:n_l    
        [w_online_3, its_online_3]= onlineLMS(A_3, y_t ,Lambdas(index_lambda), E_threshold, maxIts);
        Its_online_3(1,index_lambda) = its_online_3;
        W_online_3(:,index_lambda) = w_online_3;
    end
    
    // display iterations vs. lambda
    no_lambdaFigure = 100;
    plotIterationsVsLambda(Its_online_3, Lambdas, no_lambdaFigure);
    
    // 1.2.3.I - determine mu and Sigma of w^* coefficients
    lambda = 0.001;
    c = 3;
    dimension_end = 9;
    dimension_delta = dimension_end - dimension_start + 1;
    mu = 0;
    Sigma = 0.7;
    no_trainingsSets = 2000;
    
    WW_star = zeros(dimension_end+1,dimension_delta,no_trainingsSets);
    
    for index_trainingsSet=1:no_trainingsSets
        // determine new trainingsset
        [x_t,t]=generateTrainingsSet(x,y,G,mu,Sigma);
        y_t = t';
        // determine w_star for given dimensions
        W_star = trans_x_comp_w_star(x_t,y_t,lambda,dimension_start, dimension_end);
        // track all w_stars for given trainingsset
        WW_star(:,:,index_trainingsSet)=W_star;
    end
    
    //calculate mu and Sigma over given w_stars
    Variances = zeros(dimension_end+1,dimension_delta);
    Deviations = zeros(dimension_end+1,dimension_delta);
    for index_dimension=1:dimension_delta
        // generate correct matrix for specific dimension
        dim_w = dimension_start+index_dimension;
        W_dim_trainings = WW_star(1:dim_w,index_dimension,:);
        W_trainings = reshape(W_dim_trainings,dim_w,no_trainingsSets)';
        // matlab: For matrix input X, where each row is an observation, and each column is a variable, 
        // cov(X) is the covariance matrix. diag(cov(X)) is a vector of variances for each column, 
        // and sqrt(diag(cov(X))) is a vector of standard deviations
        vec_variance = diag(cov(W_trainings));
        vec_deviation = sqrt(vec_variance);
        Variances(1:dim_w,index_dimension) = vec_variance;
        Deviations(1:dim_w,index_dimension) = vec_deviation;        
    end
    
    // 1.2.3.II - plot for x^*=2 in the medium quadric error dimensions for f_{w^*}(x^*)
    [x_2,y_2]=generateXY(2,2,1,G);
    YY_star = zeros(dimension_delta,no_trainingsSets);
    for index_trainingsSet=1:no_trainingsSets
        W_star = WW_star(:,:,index_trainingsSet);
        Y_star = createPolynomValuesW(x_2,W_star); % column-vector for all dimensions ? 
        YY_star(:,index_trainingsSet) = Y_star;
    end
    
    // search every row in YY_star as medium of given dimension
    y_2_trainingsset = repmat(y_2, no_trainingsSets, 1);
    E = zeros(dimension_delta,1);
    for index_dimension=1:dimension_delta
       y_star_trainingsset = YY_star(index_dimension,:)';
       
       y_2_delta = y_2_trainingsset - y_star_trainingsset;
       e_trainingsset = y_2_delta.^2;
       e = mean(e_trainingsset);
       E(index_dimension)= e;
    end
    
    // plot error E in regards to dimensionality
    no_figure_meanerror = 200;
    plotMeanError(E, dimension_start, dimension_end, no_figure_meanerror);
    
    //  1.2.3.III - calculate w^* for not pretuberated trainingsset
    mu = 0;
    Sigma = 0;
    [x_t_notPret,t_notPret]=generateTrainingsSet(x,y,G,mu,Sigma);
    y_t_notPret = t_notPret';
    W_star_notPret = trans_x_comp_w_star(x_t_notPret,y_t_notPret,lambda,dimension_start, dimension_end);
    
end

function [x,y]=generateXY(x_start,x_end,x_interval,G)
    x = x_start:x_interval:x_end;
    y = 2.*x.^2-G.*x+1;
end

function [x_t,t] = generateTrainingsSet(x,y,G,my,Sigma)
    // Y=grand(m,n,'nor',Av,Sd) generates random variates from the normal distribution with mean Av (real) and standard deviation Sd (real >= 0)
    x_t = x(1:6:end);
    
    [m,n] = size(x_t);
    // Scilab
    noise = grand(m,n,'nor',my, Sigma);
    // Matlab
    //noise = normrnd(my, Sigma,m,n);
    // matlab < 2014:
    //noise = randn(m,n)*Sigma + my;
    y_t = 2*x_t.^2-G*x_t+1;
    t = y_t + noise;    
end

function plotData(x,y,x_t,t,y_star, y_star_II)
    clf;
    //set(gca(),"auto_clear","off");
    plot(x,y,'ro-',x_t,t,'bo-',x,y_star,'go-',x,y_star_II,'co-');
    //set(gca(),"auto_clear","on");    
end

function plotDataY(x,Y, mode, color)
    plotColors = ['r','g','b','c','m','y'];
    [m,n] = size(Y);
    [i,j] = size(plotColors);
    clf;
    set(gca(),"auto_clear","off");
    //hold on;%DAN
    for index=1:m
        if(nargin<4)
            plotArg = strcat([plotColors(mod(index-1,j)+1),mode]);
        else
            plotArg = strcat([color,mode]);
        end
        //plot(x,Y(index,:),'ro-');
        plot(x,Y(index,:),plotArg);
    end
    //hold off;%DAN
    set(gca(),"auto_clear","on");    
end

function plotIterationsVsLambda(Iterations, Lambdas, no_figure)
    figure(no_figure);
        plotArg = '-r';
    plot(Lambdas,Iterations,plotArg);
    title('Iterations over Lambdas');
    xlabel('Lambda');
    ylabel('Iterations');
end

function plotLinearRegressionResults(x, y, t, y_online, y_star, dimension, no_figure)
    figure(no_figure);
    //hold on;
    set(gca(),"auto_clear","off");
    
    titleStr = strcat('linear regression result for ', int2str(dimension),' dimensional function f(x)');
    
    title(titleStr);
    xlabel('x');
    ylabel('f(x)');
    
    plot(x,y,'-b');
    plot(x,y_online,'-r');
    plot(x,y_star,'-g');
    
    str_org = 'y_{original}';
    str_online = 'y_{online}';
    str_star = 'y_{star}';
    legend(str_org, str_online, str_star, 'Location', 'SouthEast');
        
    //hold off;
    set(gca(),"auto_clear","on"); 
end

function plotMeanError(E, dimension_start, dimension_end, no_figure)
    figure(no_figure);
    
    titleStr = strcat('mean error of repititive y^* calculations over dimensions');
    
    title(titleStr);
    xlabel('dimension');
    ylabel('mean error');

    x = dimension_start:dimension_end;
    y = E';
    plot(x,y,'-r');

end


function A = createA(x,d)
    // d ... dimension of polynom
    [m,n] = size(x);
    A = zeros(d,n);
    // transform x into a polynom of degree d
    for index=1:d+1
        A(index,:) = x.^(index-1);
    end    
end

function w_star = compute_w_star(A,y,lambda)
    // create pseudo inverse and compute Aw=b
    // see UNDERSTANDING MACHINE LEARNING, page 94ff
    AAT = A*A';
    b = A*y;
    succ = 0;
    A_plus = [];
    while ~succ
        try
            A_plus = inv(AAT);
            succ = 1;
        catch
            // check if above is not invertible
            [m,n] = size(AAT);
            AAT = lambda*eye(m,n) + AAT;
        end
    end
    w_star = A_plus*b;
end

function y = createPolynomValues(x,w)
    [m,n]=size(x);
    I=ones(m,n);
    y = w(1).*I;
    [m2,n2] = size(w);
    pol_degree = m2 - 1;
    for index=1:pol_degree
        y = y+w(index+1).*x.^index;
    end
end

function Y = createPolynomValuesW(x,W)
    [m,n] = size(W);
    [i,j] = size(x);
    Y = zeros(n,j);
    for index=1:n
        Y(index,:) = createPolynomValues(x,W(:,index));
    end
end

function W_star = trans_x_comp_w_star(x, y, lambda, dimension_start, dimension_end)
    // w_star is a column vector
    // W_star is a list of column vectors
    A_i = createA(x,dimension_start);
    W_star = compute_w_star(A_i,y,lambda);
    for dimension_i=dimension_start+1:dimension_end
        A_i = createA(x,dimension_i);
        // add zero line at the bottom
        [m,n] = size(W_star);
        W_star = cat(1,W_star,zeros(1,n));
        W_star = cat(2,W_star,compute_w_star(A_i,y,lambda));
    end
end

function [w_t, its] = onlineLMS(A,t,lambda, E_threshold, maxIts)
    // implement online learn from formula    
    //w(t+1) = w(t) + lambda* (t(i) - o(i)) * x(i);    
    
    [m,n] = size(A);
    w_t = zeros(m,1);
    E_tm1 = intmax;
    E_t = intmax/2;
    its =0;
    while abs(E_t - E_tm1) > E_threshold && its < maxIts
    //while abs(E_tm1/E_t) > treshold_E_ratio && its < maxIts
        E_tm1 = E_t;
        E_t=0;
        for i_index=1:n
            x_i = A(:,i_index);
            t_i = t(i_index);
            o_i = w_t'*x_i;
            // calculate cost
            E_t = E_t + (t_i-o_i)^2;
            // update w
            w_t = w_t + lambda*(t_i-o_i)*x_i;
        end
        its = its + 1;
    end
end

