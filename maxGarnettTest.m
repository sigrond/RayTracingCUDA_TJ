function [ ref_indx1, ref_indx2 ] = maxGarnettTest( fraction )


        n = 1.4678;    
        nm = 2.4645;
                
   
f = fraction;
    E = n^2;
    Em = nm^2;

    num = 3 * f * ( E - Em ) / ( E + 2 * Em );
    denom = 1 - f * ( E - Em) / ( E + 2 * Em );

    Eavg = Em * ( 1 + num ./ denom );

    ref_indx1 = sqrt(Eavg);

    
    
        n = 1.4724;        
        nm = 2.4581;

f = fraction;

    E = n^2;
    Em = nm^2;

    num = 3 * f * ( E - Em ) / ( E + 2 * Em );
    denom = 1 - f * ( E - Em) / ( E + 2 * Em );

    Eavg = Em * ( 1 + num ./ denom );

    ref_indx2 = sqrt(Eavg);

 
end
