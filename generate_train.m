clear;close all;

size_input = 20;
size_label = 20;
stride = 20;

%% scale factors
scale = 4;

%% initialization
data = zeros(size_input, size_input, 191);
label = zeros(size_label, size_label, 191);
padding = abs(size_input - size_label)/2;
count = 0;

    for i = 1 : 1
        fprintf('scale:%d,%d\n',scale,i);

        image_r = load('.\unite8\0.mat'); 
        image = image_r.label;
        [hei,wid, c] = size(image);

        for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1

                label = image(x : x+size_input-1, y : y+size_input-1, :);
                %input = add_noise(label,50/255);
                input = label;
                count=count+1;
                str1 = strcat('.\test\label\',num2str(count),'.mat');
                str2 = strcat('.\test\input\',num2str(count),'.mat');
                save(str1,'label');
                save(str2,'input')
            end
        end
    end

