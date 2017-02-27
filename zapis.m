% Automatyczna procedura do zapisywania danych po przetwarzania do pliku
try
    setup=evalin('base','setup');
    circleR=evalin('base','circleR');
    circleG=evalin('base','circleG');
    save(nazwa_nowa, 'Ipp', 'Iss', 'setup', 'theta', 'lens_center', 'circleG', 'circleR');
catch
    try           
        save(nazwa_nowa, 'Ipp', 'Iss', 'setup', 'theta', 'circleG', 'circleR');% Mniej o  zmienn¹ : lens_center
    catch
        try
            save(nazwa_nowa, 'Ipp', 'Iss', 'theta','circleG', 'circleR'); % Mniej o  zmienn¹ : setup    
        catch
            try
                save(nazwa_nowa, 'Ipp', 'Iss', 'theta'); % Mniej o  zmienne: circleR, circleG
            catch
                save(nazwa_nowa, 'Ipp', 'Iss');% Mniej o  zmienne: theta
            end
        end
    end
end


        
    
    
    