% Lineare Regression

function MyLinearRegression()
    clc, clear, close all
    x_start = 0;
    x_end = 5;
    x_interval = 0.1;
    G = 10;
    [x,y]=generateXY(x_start,x_end,x_interval,G);

    my = 0;
    Sigma = 0.7;
    [x_t,t]=generateTrainingsSet(x,y,G,my,Sigma);

    y_t = t';
    lambda = 0.0001;
    
    %%A_3 = createA(x_t,3);
    %%w_star_3 = compute_w_star(A_3,y_t,lamda);
    %%y_star_3 = createPolynomValues(x,w_star_3);
    %
    %%A_4 = createA(x_t,4);
    %%w_star_4 = compute_w_star(A_4,y_t,lamda);
    %%y_star_4 = createPolynomValues(x,w_star_4);
    %
    %%plotData(x,y,x_t,t,y_star_3,y_star_4);
    %
    dimension_start = 3;
    dimension_end = 9;
    W_star = trans_x_comp_w_star(x_t,y_t,lambda,dimension_start, dimension_end);
    Y = createPolynomValuesW(x,W_star);

    hold on %%DAN
    plotDataY(x,Y,'-', 'r'); %%DAN
    plotDataY(x_t, t,'o', 'g'); %%DAN
    %plotDataY(x_t, t,'-', 'g'); 
    hold off %%DAN

    % % test of online LMS
    %E_threshold = 0.01;
    %lambda = 0.05;
    %A_3 = createA(x_t,3);
    %w_online_3 = onlineLMS(A_3, y_t ,lambda, E_threshold);
    %[Y_online, its_online] = createPolynomValuesW(x,w_online_3);
    
    %% 1.2.2.III - test influence of lambda on convergence of online LMS
    %Lambdas = 0.001:0.001:1;
    Lambdas = 0.001:0.001:10;
    %Lambdas = 0.01:0.01:10;
    %Lambdas = 0.1:0.1:1000;
    [m_l,n_l]=size(Lambdas);
    
    A_3 = createA(x_t,3);
    E_threshold = 0.01;
    %treshold_E_ratio = 1.01;
    maxIts = 100;
    Its_online_3 = zeros(1,n_l);
    W_online_3 = zeros(4,n_l);
    for index_lambda=1:n_l    
        [w_online_3, its_online_3]= onlineLMS(A_3, y_t ,Lambdas(index_lambda), E_threshold, maxIts);
        Its_online_3(1,index_lambda) = its_online_3;
        W_online_3(:,index_lambda) = w_online_3;
    end
    
    % display iterations vs. lambda

end

function [x,y]=generateXY(x_start,x_end,x_interval,G)
    x = x_start:x_interval:x_end;
    y = 2.*x.^2-G.*x+1;
end

function [x_t,t] = generateTrainingsSet(x,y,G,my,Sigma)
    % Y=grand(m,n,'nor',Av,Sd) generates random variates from the normal distribution with mean Av (real) and standard deviation Sd (real >= 0)

    %%DAN (-> nur falls 6 random punkte)
    %%x_t = x(randperm(length(x)));
    %%x_t = sort(x_t(1:6));
    x_t = x(1:6:end);
    
    [m,n] = size(x_t);
    % Scilab
    %noise = grand(m,n,'nor',my, Sigma);
    % Matlab
    %noise = normrnd(my, Sigma,m,n);
    
    %% DAN matlab < 2014:
    noise = randn(m,n)*Sigma + my;
    y_t = 2*x_t.^2-G*x_t+1;
    t = y_t + noise;    
end

function plotData(x,y,x_t,t,y_star, y_star_II)
    clf;
    %set(gca(),"auto_clear","off");
    plot(x,y,'ro-',x_t,t,'bo-',x,y_star,'go-',x,y_star_II,'co-');
    %set(gca(),"auto_clear","on");    
end

function plotDataY(x,Y, mode, color)
    plotColors = ['r','g','b','c','m','y'];
    [m,n] = size(Y);
    [i,j] = size(plotColors);
    %clf;
    %set(gca(),"auto_clear","off");
    %hold on;%DAN
    for index=1:m
        if(nargin<4)
            plotArg = strcat([plotColors(mod(index-1,j)+1),mode]);
        else
            plotArg = strcat([color,mode]);
        end
        %plot(x,Y(index,:),'ro-');
        plot(x,Y(index,:),plotArg);
    end
    %hold off;%DAN
    %set(gca(),"auto_clear","on");    
end

function A = createA(x,d)
    % d ... dimension of polynom
    [m,n] = size(x);
    A = zeros(d,n);
    % transform x into a polynom of degree d
    for index=1:d+1
        A(index,:) = x.^(index-1);
    end    
end

function w_star = compute_w_star(A,y,lambda)
    % create pseudo inverse and compute Aw=b
    % see UNDERSTANDING MACHINE LEARNING, page 94ff
    AAT = A*A';
    b = A*y;
    succ = 0;
    A_plus = [];
    while ~succ
        try
            A_plus = inv(AAT);
            succ = 1;
        catch
            % check if above is not invertible
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
    % w_star is a column vector
    % W_star is a list of column vectors
    A_i = createA(x,dimension_start);
    W_star = compute_w_star(A_i,y,lambda);
    for dimension_i=dimension_start+1:dimension_end
        A_i = createA(x,dimension_i);
        % add zero line at the bottom
        [m,n] = size(W_star);
        W_star = cat(1,W_star,zeros(1,n));
        W_star = cat(2,W_star,compute_w_star(A_i,y,lambda));
    end
end

function [w_t, its] = onlineLMS(A,t,lambda, E_threshold, maxIts)
    % implement online learn from formula    
    %w(t+1) = w(t) + lambda* (t(i) - o(i)) * x(i);    
    
    [m,n] = size(A);
    w_t = zeros(m,1);
    E_tm1 = intmax;
    E_t = intmax/2;
    its =0;
    while abs(E_t - E_tm1) > E_threshold && its < maxIts
    %while abs(E_tm1/E_t) > treshold_E_ratio && its < maxIts
        E_tm1 = E_t;
        E_t=0;
        for i_index=1:n
            x_i = A(:,i_index);
            t_i = t(i_index);
            o_i = w_t'*x_i;
            % calculate cost
            E_t = E_t + (t_i-o_i)^2;
            % update w
            w_t = w_t + lambda*(t_i-o_i)*x_i;
        end
        its = its + 1;
    end
end

