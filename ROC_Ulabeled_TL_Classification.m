% Spyros Kasapis, Student PhD
% Dr. Nickolas Vlahopoulos, Advisor
% Geng Zhang, MES
% College of Engineering, Naval Architecture and Marine Engineering
% University of Michigan, Ann Arbor, MI
% September 4th, 2020
% Paper: "Using ROC and Unlabeled Data for Increasing Low-Shot TransferLearning Classification Accuracy"

clc;
clear all;

% General Info
Pretrain = false; % Use saved weight matrix if only working on Statistics Part
NumberOfClasses = 50; global NumberOfClasses 
TrainPicturesInClass = 40; global TrainPicturesInClass  
ValPicturesInClass = 10; global ValPicturesInClass
Epochs = 500;

% Read in and Prepare Training Dataset
TrainFile = 'features/NoIR/ResNet10SgmTrain.hdf5'; % Input is a 512 long vector X number of images
TrainingFeatures = hdf5read(TrainFile,'all_feats');
TrainingFeatures = TrainingFeatures';
TrainingCount = hdf5read(TrainFile,'count'); % Datasets as given by low-shot-shrink-hallucinate code have extra entries
TrainingFeatures = TrainingFeatures(1:TrainingCount,:); % We get rid of them here

% Read in and Prepare Validation Dataset
ValFile = 'features/NoIR/ResNet10SgmVal.hdf5';
ValFeatures = hdf5read(ValFile,'all_feats');
ValFeatures = ValFeatures';
ValCount = hdf5read(ValFile,'count');
ValFeatures = ValFeatures(1:ValCount,:);

% Pre-Train Feature Normalization
TrainingFeatures = Norm(TrainingFeatures); 
ValFeatures = Norm(ValFeatures); 

% Construct the Target Matrix
Target = zeros(2*NumberOfClasses*TrainPicturesInClass,NumberOfClasses); % Target Initialization
for Class = 1:NumberOfClasses;
    ColumnTarget = zeros(2*NumberOfClasses*TrainPicturesInClass,1);
    StartingIndex = (Class-1)*TrainPicturesInClass+1;
    EndingIndex = (Class-1)*TrainPicturesInClass+TrainPicturesInClass;
    ColumnTarget(StartingIndex:EndingIndex,1) = 1.0;
    Target(:,Class) = ColumnTarget;  
end
Target(NumberOfClasses*TrainPicturesInClass+1:2*NumberOfClasses*TrainPicturesInClass,:) = -0.2;


%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%

if Pretrain == false
    Weights = zeros(512,50); % Weight initialization
    for Class = 1:NumberOfClasses; % Train class by class
        clc; Class
        rng(1)
        ClassWeights = rand(512,1)*0.1; % Class Weights Randomization
        for e = 1:Epochs; % Calculate the loss function 
            % Gradient Calculation
            Error = TrainingFeatures*ClassWeights-Target(:,Class);
            Loss = exp(Error.^2);
            LossDerivative = 2*Error.*Loss;
            ClassGradients = (LossDerivative'*TrainingFeatures)';
            ClassGradients = (ClassGradients/norm(ClassGradients))/1; % see what happens if there is a decrease in the grad magnitude
            % Determine Learning Rate
            Flag = 1;
            TotalLoss = norm(Loss);
            LearningRate = 0.01;
            while (Flag == 1);
                TempClassWeights = ClassWeights-LearningRate*ClassGradients;
                Error = TrainingFeatures*TempClassWeights-Target(:,Class);
                Loss = exp(Error.^2);
                if (norm(Loss)<TotalLoss);
                    TotalLoss = norm(Loss);
                    LearningRate = LearningRate*2.0;
                else
                    Flag = 0;
                end
            end
            ClassWeights = ClassWeights-0.5*LearningRate*ClassGradients; % Gradient Step
        end
        Weights(:,Class) = ClassWeights; % Put Class Weight in Position
    end
else
    Weights = load('Weights50R50I.mat');
    Weights = Weights.Weights;
end


%%%%%%%%%%%%%%%% STATISTICS \ EVALUATION %%%%%%%%%%%%%%%%%%%%

% Get Score Matrices
TrainScores = TrainingFeatures*Weights; % Train Dataset
ValScores = ValFeatures*Weights; % Val Dataset

% Find Threshold Values
[MaxTrainScoreValues,MaxTrainScoreIdxs] = max(TrainScores,[],2);
Thresholds = zeros(NumberOfClasses,1);
for Class = 1:NumberOfClasses;
    StartingIndex = (Class-1)*TrainPicturesInClass+1;
    EndingIndex = Class*TrainPicturesInClass;
    CorrectlyClassifiedPictures = find(MaxTrainScoreIdxs(StartingIndex:EndingIndex,1)==Class);
    if(min(size(CorrectlyClassifiedPictures))>0);
        CorrectClassValue = TrainScores(StartingIndex:EndingIndex,Class);
        Thresholds(Class) = min(CorrectClassValue(CorrectlyClassifiedPictures));
        Thresholds(Class) = ApplyROC(MaxTrainScoreValues,MaxTrainScoreIdxs,Class,Thresholds(Class));
    end
    % Stat Function 1
    ClassThresholdGraphs(StartingIndex,EndingIndex,TrainScores,MaxTrainScoreIdxs,MaxTrainScoreValues,Class,Thresholds(Class))
end

% Count Correct Relevant
Fails = 0;
Correct = 0;
[MaxValScoreValues,MaxValScoreIdxs] = max(ValScores,[],2); 
for Class = 1:NumberOfClasses;
    StartingIndex = (Class-1)*ValPicturesInClass + 1;
    EndingIndex = Class*ValPicturesInClass;
    ClassValScores = ValScores(StartingIndex:EndingIndex,:);
    ClassMaxValScoreIdxs = MaxValScoreIdxs(StartingIndex:EndingIndex,1);
    ClassMaxValScoreValues = MaxValScoreValues(StartingIndex:EndingIndex,1)';
    ClassScores(Class) = 0;
    for Pic = 1:ValPicturesInClass;
        if ClassMaxValScoreIdxs(Pic) == Class;
            ClassScores(Class) = ClassScores(Class)+1;
            Correct = Correct + 1;
            if ClassMaxValScoreValues(Pic) < Thresholds(Class);
                ClassScores(Class) = ClassScores(Class)-1;
                Fails = Fails + 1;
            end
        end
    end
end
RelevantAccuracy = (sum(ClassScores)/(NumberOfClasses*ValPicturesInClass))*100;

% Count Correct Irrelevant
[MaxIrrelScoreValues,MaxIrrelScoreIdxs] = max(ValScores(501:1000,:),[],2);
IncorrectIrrel = 0;
for Pic = 1:500;
    if (ValScores(500+Pic,MaxIrrelScoreIdxs(Pic)) > Thresholds(MaxIrrelScoreIdxs(Pic)));
        IncorrectIrrel = IncorrectIrrel+1;
    end
end
IrrelevantAccuracy = (1.0 - (IncorrectIrrel/500))*100;

% Prints
Print1 = [RelevantAccuracy,IrrelevantAccuracy]; formatSpec1 = 'Relevant Accuracy: %4.2f \nIrrelevant Accuracy: %8.3f \n';
fprintf(formatSpec1,Print1)
Print2 = [round(Correct),round(Fails)]; formatSpec2 = 'Correct Relevant Pictures: %4.2f \nIncorrect Relevant Pictures: %8.3f \n';
fprintf(formatSpec2,Print2)
% Paper Plots
% Stat Function 2
ValThreshGraph(MaxValScoreValues,MaxValScoreIdxs,Thresholds,ValPicturesInClass)
% Stat Function 3
TrainThreshGraph(MaxTrainScoreValues,MaxTrainScoreIdxs,Thresholds,TrainPicturesInClass)


%% Functions

%%%%%%%%%%%%%%%%%%%% Auxiliary Functions %%%%%%%%%%%%%%%%%%%%%%

% Normalization
function NewMatrix = Norm(Matrix) % Get feature values from 0 to 1
    Min = min(Matrix,[],2); % Training Minimum Value
    Max = max(Matrix,[],2); % Training Maximum Value
    Range = Max-Min; % Training Range
    NewMatrix = (Matrix-Min)./Range; 
end

% Threshold Calculation
function NewThreshold = ApplyROC(MaxTrainScoreValues,MaxTrainScoreIdxs,Class,OldThreshold)
    global NumberOfClasses; global TrainPicturesInClass    
    % Relevant
    StartingIndex = (Class-1)*TrainPicturesInClass+1;
    EndingIndex = Class*TrainPicturesInClass;
    RelevantProbabilities = MaxTrainScoreValues(StartingIndex:EndingIndex);
    RelevantIndexes = MaxTrainScoreIdxs(StartingIndex:EndingIndex);
    RelevantProbabilities(find(RelevantIndexes ~= Class)) = []; % Get rid of misclassified
    [RelevantProbabilities,SortedRelIndexes] = sort(RelevantProbabilities);
    for R = 1:size(RelevantProbabilities,1)
        RelNames{R} = strcat('R',num2str(SortedRelIndexes(R)));
    end
    RelNames = RelNames';
    % Irrelevant
    IrrelevantProbabilities = MaxTrainScoreValues(NumberOfClasses*TrainPicturesInClass+1:end);
    IrrelevantIndexes = MaxTrainScoreIdxs(NumberOfClasses*TrainPicturesInClass+1:end);
    IrrelevantProbabilities(find(IrrelevantIndexes ~= Class)) = [];
    if isempty(IrrelevantProbabilities)
        IrrelevantProbabilities = -1;
    end
    [IrrelevantProbabilities,SortedIrrIndexes] = sort(IrrelevantProbabilities);
    for I = 1:size(IrrelevantProbabilities,1)
        IrrelNames{I} = strcat('I',num2str(SortedIrrIndexes(I)));
    end
    IrrelNames = IrrelNames';
    % Put the lists together and the cell arrays together and then start
    Probabilities = [IrrelevantProbabilities ; RelevantProbabilities];
    Names = [IrrelNames ; RelNames];
    [SortedProbabilities,SortedIndexes] = sort(Probabilities);
    for S = 1:size(Names,1)
        SortedNames{S} = Names{SortedIndexes(S)};
    end
    SortedNames = SortedNames';
    ROCGraph = false;
    if ROCGraph == true
        if Class == 8 | Class == 20 | Class == 45 | Class == 46 | Class == 47
            [X,Y,Thresholds,AUC] = ROCurve(SortedNames,SortedProbabilities,Class);
            AUC
            plot(X,Y)
            hold on
        end
        x = linspace(0,1,5);
        y = x;
        plot(x,y,'LineWidth',1)
        xlabel('False Relevant Rate') %Negative
        ylabel('True Relevant Rate') %Positive
        title('Receiver Operating Characteristic Curve for Select Classes')
        legend('IR Human (C8)','Binoculars (C20)','Chess Board (C45)','Chimp (C46)','Chopsticks (C47)','Random Guess','Location','southeast','FontSize',10)
    end
    % Sanity Check
    if not(size(SortedNames,1) == size(SortedProbabilities,1)) error('Spyros Error: Size Missmatch'), end;
    % Determine Lowest Threshold (Before Lowest Relevant)
    for L = 1:size(SortedNames,1)
        if SortedNames{L}(1) ~= SortedNames{L+1}(1)
            StartingThreshold = (SortedProbabilities(L) + SortedProbabilities(L+1))/2;
            StartingBound = L;
            break
        end
    end
    % Determine Ending Threshold (After Highest Irrelevant)
    for H = 1:size(SortedNames,1)
        if SortedNames{size(SortedNames,1)-H}(1) ~= SortedNames{size(SortedNames,1)-H+1}(1)
            EndingThreshold = (SortedProbabilities(size(SortedNames,1)-H) + SortedProbabilities(size(SortedNames,1)-H+1))/2;
            EndingBound = size(SortedNames,1)-H+1;
            break
        end
    end
    % Threshold Determination
    if StartingThreshold == EndingThreshold
        NewThreshold = StartingThreshold;
    else
        NewThreshold = OldThreshold; % Addition for tuning
        Index = StartingBound;
        OldCorrect = 0;
        while Index < EndingBound 
            % Evaluate
            CorrectIrrel = 0; IncorrectIrrel = 0;
            CorrectRel = 0; IncorrectRel = 0;
            % Irrelevant
            for I = 1:Index
                if SortedNames{I}(1) == 'I'
                    CorrectIrrel = CorrectIrrel + 1;
                else
                    IncorrectRel = IncorrectRel + 1;
                end
            end
            % Relevant
            for R = (Index+1):size(SortedNames,1)
                if SortedNames{R}(1) == 'R'
                    CorrectRel = CorrectRel + 1;
                else
                    IncorrectIrrel = IncorrectIrrel + 1;
                end
            end
            Correct = CorrectRel + CorrectIrrel;
            Incorrect = IncorrectRel + IncorrectIrrel;
            if Correct+Incorrect ~= size(SortedNames,1) error('Spyros Error: Size Missmatch'), end;
            if Correct > OldCorrect
                NewThreshold = (SortedProbabilities(Index) + SortedProbabilities(Index+1))/2;
                OptimalIndex = Index+0.5;
                OldCorrect = Correct;
            end
            Index = Index + 1;
        end
    end
end

%%%%%%%%%%%%%%%%%%%% Statistics Functions %%%%%%%%%%%%%%%%%%%%%%

% Individual Class Threshold
function ClassThresholdGraphs(StartingIndex,EndingIndex,TrainScores,MaxTrainScoreIdxs,MaxTrainScoreValues,Class,Threshold)
    ClassTrainScores = TrainScores(StartingIndex:EndingIndex,:); % GRAPHING CHANGE
    TopIndexes = MaxTrainScoreIdxs(StartingIndex:EndingIndex,1); % GRAPHING CHANGE
    TopValues = MaxTrainScoreValues(StartingIndex:EndingIndex,1); % GRAPHING CHANGE
    fig = figure(Class)
    % Training
    for Pic = 1:size(ClassTrainScores,1)
        if TopIndexes(Pic) == Class
            plot(ClassTrainScores(Pic,:),'Color',[0.0667 0.0667 0.0667],'LineWidth',0.05)
            hold on
            plot(Class,ClassTrainScores(Pic,Class),'.g')
        else
            plot(ClassTrainScores(Pic,:),'r')
            hold on
            plot(TopIndexes(Pic),ClassTrainScores(Pic,TopIndexes(Pic)),'*r')
        end
    end
    plot(Class,Threshold,'db')
    xlabel('Class')
    ylabel('Probability')
    str = ['Probabilities for the fourty images of Class ',num2str(Class)];
    title(str)
    hold off
    saveas(fig,fullfile('features/graphs/JPEGs', ['Class',num2str(Class),'Probabilities']), 'jpeg')
end 

% Train Thresholds Visualization Graph
function ValThreshGraph(MaxValScoreValues,MaxValScoreIdxs,Thresholds,ValPicturesInClass)
    figure(999)
    hold on
    plot(NaN,NaN,'o','Color',[0.4940 0.1840 0.5560],'DisplayName','Relevgant');
    plot(NaN,NaN,'o','Color',[0 0.4470 0.7410],'DisplayName','Irrelevant');
    plot(NaN,NaN,'Color',[0.4660 0.6740 0.1880],'LineWidth',3,'DisplayName','Threshold');
    MaxValScoreValuesRel = MaxValScoreValues(1:size(MaxValScoreValues,1)/2);
    MaxValScoreValuesIrrel = MaxValScoreValues((size(MaxValScoreValues,1)/2+1):end);
    MaxValScoreIdxsRel = MaxValScoreIdxs(1:size(MaxValScoreIdxs,1)/2);
    MaxValScoreIdxsIrrel = MaxValScoreIdxs((size(MaxValScoreIdxs,1)/2+1):end);
    for Class = 1:size(Thresholds,1)
        plot(Class,MaxValScoreValuesRel(((Class-1)*ValPicturesInClass+1):(Class*ValPicturesInClass)),'o','Color',[0.4940 0.1840 0.5560],'MarkerSize',4.5,'HandleVisibility','off')
        plot(linspace(Class-0.45,Class+0.45),Thresholds(Class)*ones(size(linspace(1,25),2)),'Color',[0.4660 0.6740 0.1880],'LineWidth',3,'HandleVisibility','off')
        for i = 1:size(MaxValScoreIdxsIrrel)
            if MaxValScoreIdxsIrrel(i) == Class
                plot(Class,MaxValScoreValuesIrrel(i),'o','Color',[0 0.4470 0.7410],'MarkerSize',4.5,'HandleVisibility','off')
            end
        end
    end
    xlim([-1,51])
    xlabel('Class')
    ylabel('Probability')
    title('Validation Statistics')
    legend('Relevant','Irrelevant','Thresholds')
    hold off
end

% Train Thresholds Visualization Graph
function TrainThreshGraph(MaxTrainScoreValues,MaxTrainScoreIdxs,Thresholds,TrainPicturesInClass)
    figure(1000)
    hold on
    plot(NaN,NaN,'o','Color',[0.4940 0.1840 0.5560],'DisplayName','Relevant');
    plot(NaN,NaN,'o','Color',[0 0.4470 0.7410],'DisplayName','Irrelevant');
    plot(NaN,NaN,'Color',[0.4660 0.6740 0.1880],'LineWidth',3,'DisplayName','Threshold');
    MaxTrainScoreValuesRel = MaxTrainScoreValues(1:size(MaxTrainScoreValues,1)/2);
    MaxTrainScoreValuesIrrel = MaxTrainScoreValues((size(MaxTrainScoreValues,1)/2+1):end);
    MaxTrainScoreIdxsRel = MaxTrainScoreIdxs(1:size(MaxTrainScoreIdxs,1)/2);
    MaxTrainScoreIdxsIrrel = MaxTrainScoreIdxs((size(MaxTrainScoreIdxs,1)/2+1):end);
    for Class = 1:size(Thresholds,1)
        plot(Class,MaxTrainScoreValuesRel(((Class-1)*TrainPicturesInClass+1):(Class*TrainPicturesInClass)),'o','Color',[0.4940 0.1840 0.5560],'MarkerSize',4.5,'HandleVisibility','off')
        plot(linspace(Class-0.45,Class+0.45),Thresholds(Class)*ones(size(linspace(1,25),2)),'Color',[0.4660 0.6740 0.1880],'LineWidth',3,'HandleVisibility','off')
        for i = 1:size(MaxTrainScoreIdxsIrrel)
            if MaxTrainScoreIdxsIrrel(i) == Class
                plot(Class,MaxTrainScoreValuesIrrel(i),'o','Color',[0 0.4470 0.7410],'MarkerSize',4.5,'HandleVisibility','off')
            end
        end
    end
    xlim([-1,51])
    xlabel('Class')
    ylabel('Probability')
    title('Training Statistics')
    legend('Relevant','Irrelevant','Thresholds')
    hold off
end

% Create ROC Curve using Matlab's function
function [X,Y,Thresholds,AUC] = ROCurve(Labels,Scores,Class)
    Posclass = 'R';
    for x = 1:size(Labels,1)
        Labels(x) = {string(Labels(x)){1}(1)};
    end
    figure(Class)
    [X,Y,Thresholds,AUC] = perfcurve(Labels,Scores,Posclass);
end