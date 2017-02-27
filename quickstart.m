load('Summed.mat');
try
[ Pk, PCCD ]=BorderRecognition(temp, [Pk PCCD])
catch
    [ Pk, PCCD ]=BorderRecognition(temp)
end
