% This program dispaly individual pixel's spectrum
% Last modified by Xin Liu 02-25-2015

% Select the CSV file
clear;
[csv_filename, DefaultPathname] = uigetfile( {'*.csv','csv Files'}, 'Pick a csv file');
fid_Name = strcat(DefaultPathname,csv_filename);

% Read data from csv file directly into rawdata
rawdata = csvread(fid_Name);

%col 1 - module_number = C{1};
%col 2 - time_stamp = C{2};
%col 3 - photon_index = C{3};
%col 4 - pixel_index = C{4};
%col 5 - energy channel = C{5};
%col 6 - penergy = C{6};
%col 7 - time_detect = C{7};
%col 8 - ptime_detect = C{8};

cindex = rawdata(:,5)+1; % channel index [1:4096]
pindex = rawdata(:,4);   % pixel index [1:484]
[row, col] = size(cindex);
pix_energy = zeros(484,4096); % initialize the counts for each pixel
% Accumulate counts for each pixel
for ii=1:row
    pix_energy(pindex(ii),cindex(ii))= pix_energy(pindex(ii),cindex(ii))+1;
end

% Dispaly individual pixel spectrum
figure;
for ii = 1:49
    for jj = 1:10
        subplot(2, 5, jj);
        plot(1:4096, pix_energy(jj+(ii-1)*10,:));
        grid on;
        title(['Pixel #',num2str(jj+(ii-1)*10)]);
    end
    pause
end
