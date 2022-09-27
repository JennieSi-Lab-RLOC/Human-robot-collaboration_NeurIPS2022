%%
classdef dHDP_multi
    properties
    
        timeStep;                     % scalar variable for time step
        iterationStep;                % scalar variable for policy iteration
        
        nState;
        nAction;
        nSA;
        
        blocks;
        blockNum;
        steps;
        
        sampleLength;
        blockSampleLength;
        
        outputLimit;
        
    end
    methods
        function obj = dHDP_multi(nstate, naction, annhidden, cnnhidden, Bias)
            obj.nState = nstate;
            obj.nAction = naction;
            obj.nSA = nstate + naction;
           
            obj.iterationStep = 1;

%             4 phases
            obj.blockNum = 4;
            
            obj.outputLimit = [0.1, 0.02, 0.1, 0.02];
            
%             obj.sampleLength = 40;
            
            for i = 1 : obj.blockNum
                obj.blocks{i} =  dHDP(nstate, naction, annhidden, cnnhidden,i, Bias);
            end
        end 
        
        % set controlType to one for discrete control problem
        function mql = setControlType(mql,controlType)
            for i = 1 : mql.blockNum
                mql.blocks{i} =  mql.blocks{i}.setControlType(controlType);
            end
        end
   
        function mql = addSample(mql, state, action, activeBlock)
%             for i = 1 : mql.blockNum
%                 mql.blocks{i} = mql.blocks{i}.addSample(state(i,:), action(i,:));
%             end
              mql.blocks{activeBlock} = mql.blocks{activeBlock}.addSample(state(activeBlock,:), action(activeBlock,:));
              mql.blocks{activeBlock}.sampleLength
              mql.blocks{activeBlock}.iterationStep
        end
        
        function [action, value, mql] = getAction(mql, state, activeBlock)
            if activeBlock > 0 % get action for single phase (activeBlock = 1~4)
                for i = 1 : mql.blockNum
                    if i == activeBlock
                        % limit the range of the output
                        [actionSeq, valueSeq, mql.blocks{i}] = mql.blocks{i}.getAction(state(i,:));
    %                     actionSeq = actionSeq * 0.1;
    %                     actionSeq(actionSeq > mql.outputLimit(i)) =  mql.outputLimit(i);
    %                     actionSeq(actionSeq < -mql.outputLimit(i)) =  -mql.outputLimit(i);
                        action(i,:) = actionSeq;
                        value(i,:) = valueSeq;
                    else
                        action(i,:) = zeros(1, mql.nAction);
                        value(i,:) = 0;
                    end       
                end
            else % get action for all phase (activeBlock = 0)
                for i = 1 : mql.blockNum
                        [actionSeq, valueSeq, mql.blocks{i}] = mql.blocks{i}.getAction(state(i,:));
                        action(i,:) = actionSeq;
                        value(i,:) = valueSeq;  
                end
            end            
        end
        
        % get action from the j-th policy
        function [action, mql] = getActionFromPrePolicy(mql, state, activeBlock, j)
            for i = 1 : mql.blockNum
                if i == activeBlock
                    % limit the range of the output
                    if size(state,1) > 1
                        actionSeq = mql.blocks{i}.getActionFromPrePolicy(state(i,:), j);
                    else
                        actionSeq = mql.blocks{i}.getActionFromPrePolicy(state, j);
                    end
%                     actionSeq = actionSeq * 0.1;
%                     actionSeq(actionSeq > mql.outputLimit(i)) =  mql.outputLimit(i);
%                     actionSeq(actionSeq < -mql.outputLimit(i)) =  -mql.outputLimit(i);
                    action(i,:) = actionSeq;
                else
                    action(i,:) = zeros(1, mql.nAction);
                end
                
            end
        end
        
        % update Q-learning agent
        function mql = update(mql, prevState, state, reinforcement,status, activeBlocks)
            for i = 1 : size(activeBlocks, 2)
                phaseID = activeBlocks(i);
                mql.blocks{phaseID} = mql.blocks{phaseID}.update(prevState(phaseID,:), state(phaseID,:), reinforcement(phaseID),status(phaseID), []);  
            end
%             mql.blocks{activeBlock} = mql.blocks{activeBlock}.updatePolicy(state(activeBlock,:));
%             mql.iterationStep = mql.iterationStep + 1;
        end
        
        % policy evaluation and policy improvement
        function mql = resetPolicy(mql, activeBlock)
            phaseID = activeBlock;
            mql.blocks{phaseID} = mql.blocks{phaseID}.resetPolicy();  
        end
        
                % policy evaluation and policy improvement
        function mql = setSampleLengthOfBlock(mql, activeBlock, new)
            phaseID = activeBlock;
            mql.blocks{phaseID} = mql.blocks{phaseID}.setSampleLength(new);  
        end
        
        function mql = resetANN(mql,activeBlock)
            phaseID = activeBlock;
            for i=1:length(phaseID)
                mql.blocks{phaseID(i)} = mql.blocks{phaseID(i)}.resetANN();
            end
        end
        
        function mql = resetNetwork(mql,activeBlock)
            phaseID = activeBlock;
            for i=1:length(phaseID)
                mql.blocks{phaseID(i)} = mql.blocks{phaseID(i)}.resetNetwork();
            end
        end
    end
end