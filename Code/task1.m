%%
clear
% Dataset
data_folder = 'Data/';
totalCorrectness = 0; 

load([data_folder,'data.mat'])
Ns = 200;

%Test Ratio
test_ratio = 0.2;
% Dimensions to reduce to
L = 100;
% PCA or MDA
MDA = false;
% K-NN or Bayes
KNN = true;
K = 3; 



face_n = face(:,:,1:3:3*Ns);
face_x = face(:,:,2:3:3*Ns);

numTrials = 9; 
for i = 1:numTrials
    clearvars -except data_folder face_n face_x KNN MDA L totalCorrectness Ns test_ratio i numTrials K 
    X = sprintf('Trial: %d', i);
    %disp(X)
    
    
    data = [];
    labels = [];

    % for all 200 faces, create a data vector (label 0 or 1 vector)
    % for the expression and nuetral face
    [m,n] = size(face_n(:,:,1));

    for subject=1:Ns
        %neutral face: label 0
        face_n_vector = reshape(face_n(:,:,subject),1,m*n);
        data = [data ; face_n_vector];
        %n_vectors = [n_vectors ; face_n_vector];
        labels = [labels 0];
        %face with expression: label 1
        face_x_vector = reshape(face_x(:,:,subject),1,m*n);
        data = [data ; face_x_vector];
        labels = [labels 1];
    end

    % can leave random - doesn't affect initialization of training data
    [data_len,data_size] = size(data);
    N = round((1-test_ratio)* data_len);
    idx = randperm(data_len);
    train_data = data(idx(1:N),:);
    train_labels = labels(idx(1:N));

    % collect sample of testing images
    test_data = data(idx(N+1:2*Ns),:);
    test_labels = labels(idx(N+1:2*Ns));

    if MDA == false
        % disp("Implementing PCA")
        newTrainingVectors = zeros(size(train_data,1),L);
        coeff = pca(train_data);

        % project our training data onto the PCA
        for i = 1:size(train_data,1)
            for j = 1:L
                newTrainingVectors(i,j) = train_data(i,:)*coeff(:,j);
            end
        end


        newTestVectors =  zeros(size(test_data,1),L);
        for i = 1:size(test_data,1)
            for j = 1:L
                newTestVectors(i,j) = test_data(i,:)*coeff(:,j);
            end
        end

        %disp("PCA done")

    else
        %disp("Implementing MDA")

        % We know MDA in task one will only be comprised of 2 classes

        % lets seperate our samples by class
        nVectors = [];
        xVectors = [];
        for i = 1:length(train_labels)
            if train_labels(i) == 0
                nVectors = [nVectors;train_data(i,:)];

            else
                xVectors = [xVectors;train_data(i,:)];

            end
        end

        % class means vectors (1x504)
        m1 = mean(nVectors);
        m2 = mean(xVectors);
        meanOv  = mean(train_data);
        % both will be 504x504
        S1 = zeros(size(nVectors,2),size(nVectors,2));
        S2 = zeros(size(xVectors,2),size(xVectors,2));

        n1 = size(nVectors,1);
        n2 = size(xVectors,1);

        for x = 1:n1
            diff = nVectors(x)-m1;
            scat = diff.'*diff;
            S1 = S1 + scat;
        end
        S1 = (1/n1).*S1;

        for x = 1:n2
            diff = xVectors(x)-m2;
            scat = diff'*diff;
            S2 = S2 + scat;
        end
        S2 = (1/n2).*S2;

        SW = S1 + S2;


        % Sum across classes (only 2)
        % SB = n1(m1-m)(m1-m)' + n2(m2-m)(m2-m)'
        SB = ( n1*(m1-meanOv).' * (m1-meanOv) ) + ( n2*(m2-meanOv).' * (m2-meanOv));

        % swi = inv(SW);
        % det(SW)
        SwiSb = pinv(SW)*SB;

        [U,S,V]=svd(SwiSb);
        proj = U(:,1);

        newTrainingVectors = []; 
        for i = 1:size(train_data,1)
            newTrainingVectors(i) = train_data(i,:)*proj;
        end
        newTrainingVectors = newTrainingVectors';
        newTestVectors = []; 
        for i = 1:size(test_data,1)
            newTestVectors(i) = test_data(i,:)*proj;
        end
        newTestVectors = newTestVectors'; 

        %disp("MDA done")
    end




    if KNN == true

        %display("Classifying with KNN rule");

        correctClassifications = 0;
        incorrectClassifications = 0;

        % we now look at our testing data image by image.
        for currentTestIndex = 1:size(newTestVectors,1)

            currentTestVector = newTestVectors(currentTestIndex,:);
            distArray = zeros(size(newTrainingVectors,1),1);

            for TI = 1:length(distArray)
                sum = 0;
                for i = 1:length(currentTestVector)
                    sum = sum + (newTrainingVectors(TI,i)-currentTestVector(i))^2;
                end
                distArray(TI) = sqrt(sum);
            end

            % we now have a distance array for the current sample/test image to
            % all training images

            % find the closest (min distance) k of the training data and the
            % indices
            [B, index] = mink(distArray, K);

            neutralNeighbors = 0;
            exprNeighbors = 0;
            trueClass = test_labels(currentTestIndex);

            for i = 1:length(index)
                %neutral face: label 0 %face with expression: label 1
                if train_labels(index(i)) == 0
                    neutralNeighbors = neutralNeighbors + 1;
                else
                    exprNeighbors = exprNeighbors + 1;
                end
            end

            % if there are more neutralNeighbors than exprNeighbors, classify the
            % sampe as neutralNeighbors
            if neutralNeighbors > exprNeighbors
                class = 0;
            elseif neutralNeighbors < exprNeighbors
                class = 1;
            end

            % compare our classification to the true class of the current
            % sample
            if trueClass == class
                correctClassifications = correctClassifications + 1;
            else
                incorrectClassifications = incorrectClassifications + 1;
            end

        end

        perc = (correctClassifications / (correctClassifications+incorrectClassifications)) * 100;
        
        totalCorrectness = totalCorrectness + perc; 
        
        X = sprintf('Classified %d images correctly out of %d \nCorrectness: %.2f %%',correctClassifications,correctClassifications+incorrectClassifications, perc);
        % disp(X)
    else


        %display("Classifying with Bayes rule");

        % lets seperate our samples by class
        nVectors = [];
        xVectors = [];
        for i = 1:length(train_labels)
            if train_labels(i) == 0
                nVectors = [nVectors;newTrainingVectors(i,:)];

            else
                xVectors = [xVectors;newTrainingVectors(i,:)];

            end
        end

        % class means vectors (1x504)
        m1 = mean(nVectors);
        m2 = mean(xVectors);
        n1 = size(nVectors,1);
        n2 = size(xVectors,1);

        % dimensions
        L = size(xVectors,2);


        sigma1 = zeros(L);
        sigma2 = zeros(L);

        for x = 1:n1
            sigma1 = sigma1 + (nVectors(x,:)-m1).'*(nVectors(x,:)-m1);
        end
        for x = 1:n2
            sigma2 = sigma2 + (xVectors(x,:)-m2).'*(xVectors(x,:)-m2);
        end

        sigma1 = sigma1/n1;
        sigma2 = sigma2/n2;

        classEst = ones(1,length(test_labels));

        
        % now lets check every test image and classify it ( check
        % afterwards) 
        for ind = 1:length(classEst)
            g1 = -1/2*log(det(sigma1)) - 1/2*(newTestVectors(ind,:)-m1)*inv(sigma1)*(newTestVectors(ind,:)-m1).';
            g2 = -1/2*log(det(sigma2)) - 1/2*(newTestVectors(ind,:)-m2)*inv(sigma2)*(newTestVectors(ind,:)-m2).';
            
            if g1 > g2
                classEst(1,ind) = 0;
            else
                classEst(1,ind) = 1;
            end
        end

            
        correctClassifications = 0;
        incorrectClassifications = 0; 
        
        for ind = 1:length(test_labels)
            if(classEst(1,ind) == test_labels(ind))
                correctClassifications = correctClassifications + 1; 
            else 
                incorrectClassifications = incorrectClassifications + 1;
            end 
        end 
    
        perc = (correctClassifications / (correctClassifications+incorrectClassifications)) * 100;
        
        totalCorrectness = totalCorrectness + perc; 
        
         X = sprintf('Classified %d images correctly out of %d \nCorrectness: %.2f %%',correctClassifications,correctClassifications+incorrectClassifications, perc);
         % disp(X)

    end
    

end 



if KNN == true 
    A = sprintf("Using K-NN rule, Neighbors (K): %d", K);
else 
   A = sprintf("Using Bayes Classifier");
end 

if MDA == true 
    B = sprintf("Implemented with MDA");
else
    B = sprintf("Implemented with PCA");
end 
    
Y = sprintf("Number of trials: %d\nAverage correctness across all trials: %.f%%", numTrials, totalCorrectness/numTrials);
display("===============================")
disp(A)
disp(B)
disp(Y)
 
