%%
clear

data_folder = 'Data/';

load([data_folder,'illumination.mat'])
[data_size,images,subjects] = size(illum);

% PCA or MDA
MDA = false;
% K-NN or Bayes
KNN = true;
K = 3;
L = 15; 

% 68 subjects
% 21 images per subject (21 different illuminations)
% size: 48x40

trainingImagesPerSubject = 17;
testingImagesPerSubject = 21-trainingImagesPerSubject;
numTrials = 2;
totalCorrectness = 0; 
for i = 1:numTrials
    
    clearvars -except data_folder data_size images subjects MDA KNN trainingImagesPerSubject testingImagesPerSubject numTrials L illum K totalCorrectness
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];
    
    for s=1:subjects
        
        idx = randperm(21);
        % pick 17 images per subject at random for training (17x68 = 1156 images)
        % remaining (21-4)=4 (4x68 = 272) for testing
        
        for i=1:trainingImagesPerSubject
            train_data = [train_data;illum(:,idx(i),s)'];
            train_labels = [train_labels s];
        end
        
        for i=1:testingImagesPerSubject
            test_data = [test_data;illum(:,idx(i+17),s)'];
            test_labels = [test_labels s];
        end
        
        
    end
    % display('Loaded in training data (17x68 = 1156 images)');
    % display('Loaded in test data (4x68 = 272 images)');
    
    
    % we now have our training and test data loaded in
    
    
    % implement PCA
    
    if MDA == false
        
        display('PCA') 
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
        
        
        % newTrainingVectors and newTestVectors now have the image vectors reduced
        % to 100 dimensions
    else
        
        
        % MDA
        
        display('MDA') 
        
        % we must seperate our data by the 68 classes/subjects
        % we know that the first 17 belong to class 1, next 17 to class 2, and so
        % on
        
        meanValues = [];
        
        for i = 1:trainingImagesPerSubject:length(train_labels)
            meanVector = mean(train_data(i:i+trainingImagesPerSubject-1,:));
            meanValues =[meanValues;meanVector];
        end
        
        meanOv = mean(meanValues);
        
        % SW is 1920x1920
        SW = zeros(length(meanOv));
        SB = zeros(length(meanOv));
        for i = 0:subjects-1
            
            start = i*trainingImagesPerSubject + 1;
            en = (i+1)*trainingImagesPerSubject;
            
            Si =  zeros(length(meanOv));
            for x = start:en
                diff = train_data(x,:)-meanValues(i+1,:);
                scat = diff.'*diff;
                Si = Si + scat;
            end
            Si = (1/trainingImagesPerSubject).*Si;
            SW = SW + Si;
            
            sb = (trainingImagesPerSubject*(meanValues(i+1,:)-meanOv).' * (meanValues(i+1,:)-meanOv));
            SB = SB + sb;
            
        end
        
        
        SwiSb = pinv(SW)*SB;
        
        [U,S,V]=svd(SwiSb);
        projs = U(:,1);
        
        for i = 2:L
            projs = [projs, U(:,i)];
        end
        
        newTrainingVectors = [];
        for i = 1:size(train_data,1)
            curr = train_data(i,:)*projs;
            newTrainingVectors = [newTrainingVectors; curr];
        end
        
        newTestVectors = [];
        for i = 1:size(test_data,1)
            curr = test_data(i,:)*projs;
            newTestVectors = [newTestVectors; curr];
        end
        
    end
    
    
    if KNN == true

        % KNN rule
        display('Classifying KNN') 
        correctClassifications = 0;
        incorrectClassifications = 0;

        for currentTestIndex = 1:length(test_labels)

            % single test vector
            currentTestVector = newTestVectors(currentTestIndex,:);
            % distance array to compare current to all trianing vectors
            distArray = zeros(size(newTrainingVectors,1),1);

            % iterate through training vectors and compute distance from each to
            % the current test vector
            for TI = 1:length(distArray)
                sum = 0;
                for i = 1:length(currentTestVector)
                    sum = sum + (newTrainingVectors(TI,i)-currentTestVector(i))^2;
                end
                distArray(TI) = sqrt(sum);
            end

            % get the lowest K distances and the corresponding indices
            [B, index] = mink(distArray, K);

            trueClass = test_labels(currentTestIndex);

            neighborFreq = zeros(1,subjects);

            for i = 1:length(index)
                neighborFreq(train_labels(index(i))) =  neighborFreq(train_labels(index(i))) + 1;
            end

            [A,closestNeighbor] = maxk(neighborFreq,1);



            if trueClass == closestNeighbor
                correctClassifications = correctClassifications + 1;
            else
                incorrectClassifications = incorrectClassifications + 1;
            end

        end
        perc = (correctClassifications / (correctClassifications+incorrectClassifications)) * 100;

        totalCorrectness = totalCorrectness + perc;

        X = sprintf('Classified %d images correctly out of %d \nCorrectness: %.2f %%',correctClassifications,correctClassifications+incorrectClassifications, perc);
        disp(X)
        
    else 
   
        % Bayes Classifier
        meanValues = [];

        for i = 1:trainingImagesPerSubject:length(train_labels)
            meanVector = mean(newTrainingVectors(i:i+trainingImagesPerSubject-1,:));
            meanValues = [meanValues;meanVector];
        end
        meanOv = mean(meanValues);

        L = size(meanVector,2);

        sigmas = zeros(L);
        for subject = 1:subjects

            sig = zeros(L);
            first = trainingImagesPerSubject*(subject-1)+1;
            last = trainingImagesPerSubject*subject;

            for vecNum = first:last
                sig = sig + (newTrainingVectors(vecNum,:)-meanValues(subject,:)).' * (newTrainingVectors(vecNum,:)-meanValues(subject,:));
            end
            sig = sig/trainingImagesPerSubject;

            sigmas(:,:,subject) = sig(:,:);
        end

        % we now have a collection of sigmas (for each subject)

        % we then look at each test image and calculate
        % -1/2*log(det(sigma1)) - 1/2*(newTestVectors(ind,:)-m1)*inv(sigma1)*(newTestVectors(ind,:)-m1).';
        % keep the maximum across 68 sigmas/classes and classify as that class


        correctClassifications = 0;
        incorrectClassifications = 0;

        for testIndex = 1:length(test_labels)
            maxMatch = -Inf;
            closestClass = -1;
            for classIndex = 1:subjects

                c = -1/2*log(det(sigmas(:,:,classIndex)));
                d = (1/2*(newTestVectors(testIndex,:)-meanValues(classIndex,:)) );
                e = pinv(sigmas(:,:,classIndex));
                f =(((newTestVectors(testIndex,:)-meanValues(classIndex,:))).');

                g = c - (d*e*f);
                %g = -1/2*log(det(sigmas(:,:,classIndex))) - (1/2*(newTestVectors(testIndex,:)-meanValues(classIndex)) )* (pinv(sigmas(:,:,classIndex))) * (((newTestVectors(testIndex,:)-meanValues(classIndex))).')
                % g = 1
                % if current g is greater than current maximum update
                if g > maxMatch
                    maxMatch = g;
                    closestClass = classIndex;
                end

            end

            trueClass = test_labels(testIndex);


            if trueClass == closestClass
                correctClassifications = correctClassifications + 1;
            else
                incorrectClassifications = incorrectClassifications + 1;
            end
        end

        perc = (correctClassifications / (correctClassifications+incorrectClassifications)) * 100;

        totalCorrectness = totalCorrectness + perc;

        X = sprintf('Classified %d images correctly out of %d \nCorrectness: %.2f %%',correctClassifications,correctClassifications+incorrectClassifications, perc);
        disp(X)
    end 
    
    
end

X = sprintf('End of Trials \nCorrectness: %.2f %%',totalCorrectness/numTrials);
disp(X)

