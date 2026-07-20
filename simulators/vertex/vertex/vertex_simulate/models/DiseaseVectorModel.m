classdef DiseaseVectorModel < handle
    %DiseaseVectorModel Class that represents the concentration of pathogeneic protein in each
    %cell
    %   This class encodes a model of pathogeneic protein
    %   production and spreading in neural tissue, as an addon to the
    %   VERTEX simulator. 
    %   The current implementation of disease vector spreading in VERTEX is
    %   based on spreading through synaptic transfer. In particular the
    %   synaptic transfer-specific part of the model by
    %   Georgiadis et. al (2018) https://doi.org/10.1371/journal.pone.0192518
    %   inspires the mechanism we use. 
    
    properties
        pC % amount of pathogeneic protein (mg) (2D circular array - it stores the historic value)
        nC % amount of normal protein (mg)
        buffer % post synaptic buffer for storing incoming changes to the protein concentration
        bufferCount % stores current position of the buffer
        pCTraceInd % stores the current position in pC
        bufferMax % the size of the buffer for incoming changes (timesteps)
        bufferMaxDV % how long to store the historic value of pC (timesteps)
        Rpp % rate of production of pathogenic protein (mg/s)
        Rpn % rate of production of normal protein (mg/s)
        Rm % rate of misfolding 
        Rcp % rate of clearance of pathogenic protein (mg/s)
        Rcn % rate of clearance of normal protein (mg/s)
        Cpn % baseline of pathogenic protein (mg)
        Cnn % baseline of normal protein (mg)
        Rsyn % rate of synaptic transmission of pathogenic protein (mg/spike)
        
    end
    
    methods
        function DVM = DiseaseVectorModel(number,DVMP, SS)
            %DiseaseVectorModel Constructs an instance of this class
            %   Model paramters are stored in DVMP.
            %   number is the number of cells in the model
            %   This class represents all cells in the model so for example: 
            %   pC stores a value for every cell. 
            %   SS is the simulation settings struct.

            
            
            % Initialise variables:
            % Circular array used to store the concentration of pathogenic
            % protein so that we can model delays. 
            DVM.pC = zeros(SS.maxDelayStepsDV,number); 
            DVM.pC(1,:) = DVMP.initpC;
            DVM.nC = DVMP.initnC .* ones(1,number);
            DVM.buffer = zeros(number, SS.maxDelaySteps);
            DVM.pCTraceInd = 1;
            DVM.bufferCount = 1;
            DVM.bufferMax = SS.maxDelaySteps;
            DVM.bufferMaxDV = SS.maxDelayStepsDV;
            
            % Store model parameters
            DVM.Rpp = DVMP.Rpp;
            DVM.Rpn = DVMP.Rpn;
            DVM.Rm = DVMP.Rm;
            DVM.Rcp = DVMP.Rcp;
            DVM.Rcn = DVMP.Rcn;
            DVM.Cpn = DVMP.Cpn;
            DVM.Cnn = DVMP.Cnn;
            DVM.Rsyn = DVMP.Rsyn;

        end
        
        function DVM = updateDiseaseVectorModel(DVM, dt)
            %updateDiseaseVectorModel Updates the state of the model at each
            %time step.
            %   Based on the equations in (Georgiadis,2018) https://doi.org/10.1371/journal.pone.0192518
            %   This function models protein production and misfolding.
            
            % check whether we have reached the end of our circular array.
             if DVM.pCTraceInd < DVM.bufferMaxDV
                 DVM.pCTraceInd = DVM.pCTraceInd+1;
                 DVM.pC(DVM.pCTraceInd,:) = DVM.pC(DVM.pCTraceInd-1,:);
             else % if we have then go back to the start. 
                 DVM.pCTraceInd = 1;
                 DVM.pC(DVM.pCTraceInd,:) = DVM.pC(DVM.bufferMaxDV,:);    
             end
            
             % proteins accumulate through production 
             % pathogenic protein accumulates (equation 1)
             DVM.pC(DVM.pCTraceInd,:) = DVM.pC(DVM.pCTraceInd,:) + DVM.Rpp.*dt;
             % normal protein accumulates (equation 2)
             DVM.nC = DVM.nC + DVM.Rpn.*dt; 
             
             % proteins misfold at a rate of b (equation 5)
             b = DVM.Rm .* DVM.pC(DVM.pCTraceInd,:) .* DVM.nC;
             % pathogenic protein increases by by
             DVM.pC(DVM.pCTraceInd,:) = DVM.pC(DVM.pCTraceInd,:) + b.*dt;
             % normal protein decreases by b
             DVM.nC = DVM.nC - b.*dt; 
             
             % proteins clear at a rate of qp,qn so as to get back to
             % baseline concentrations Cpn, Cnn
             % equation 3 for pathogenic protein clearance
             qp = DVM.Rcp .* log(1+ (exp(1) - 1).*(DVM.pC(DVM.pCTraceInd,:)./DVM.Cpn));
             % equation 3 for normal protein clearance
             qn = DVM.Rcn .* log(1+ (exp(1) - 1).*(DVM.nC./DVM.Cnn));
             % equation 4 for pathogenic protein clearance
             DVM.pC(DVM.pCTraceInd,:) = DVM.pC(DVM.pCTraceInd,:) - qp.*dt;
             % equation 4 for normal protein clearance
             DVM.nC = DVM.nC - qn.*dt;            
        end
        
        % applied in simulate/simulateParallel function after each spike (action potential).
        % the spike causes the synapse to open which causes the
        % transmission of pathogenic protein. This function is called to
        % remove the protein from the presynaptic cell at a rate of Rsyn
        % per spike.
        function DVM = updatePresynapticCellsAfterSpike(DVM, presynIDs)
            % computes equation 11
            DVM.pC(DVM.pCTraceInd,presynIDs) = DVM.pC(DVM.pCTraceInd,presynIDs) - DVM.pC(DVM.pCTraceInd,presynIDs).*DVM.Rsyn;
        end
        
        % updates the buffer for axonal transmission delay
        % this function updates the circular array that stores the pathogenic protein
        % concentration in each cell using the current value in the buffer.
        % 
        function DVM = updateBuffer(DVM)
            DVM.pC(DVM.pCTraceInd,:) = DVM.pC(DVM.pCTraceInd,:) + DVM.buffer(:, DVM.bufferCount)';
            DVM.buffer(:, DVM.bufferCount) = 0;
            DVM.bufferCount = DVM.bufferCount + 1;
            
            if DVM.bufferCount > DVM.bufferMax
                DVM.bufferCount = 1;
            end
        end
        
        % updates the post synaptic buffer with the synaptically
        % transmitted concentration.
        function DVM = bufferVectorFlow(DVM, postNeuronIDs,delays,~,presynPrC)
            % postNeuronIDs: contains list of post synaptic neuron IDs on
            % which to perform the update.
            % delays: contains the delay to apply (precalculated based on
            % distance between pre and post synaptic cells)
            % presynPrC: the concentration of the pathogenic protein in the
            % presynaptic cell.
            % for each post synaptic cell
            for i = 1:length(postNeuronIDs)
                % this partially computes equations 9 and 10
                DVM.buffer(postNeuronIDs(i),delays(i)) =  DVM.buffer(postNeuronIDs(i),delays(i)) + (presynPrC(i).*DVM.Rsyn)./length(postNeuronIDs);
            end
        end
        
    end
end

