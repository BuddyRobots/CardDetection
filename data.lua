require 'torch'
require 'util'
require 'image'
require 'gnuplot'
require 'lfs'

-- mean_file = assert(io.open("models/mean", "r"))
-- train_mean = tonumber(mean_file:read())
img_size = 28

function load_data()
	local type_idx = 1

	local imgs_count = 0

	for sub_folder in lfs.dir(type_str .. "_set") do
		if (sub_folder ~= "." and sub_folder ~= "..") then
			for image_file in lfs.dir(type_str .. "_set/" .. sub_folder) do
				if (image_file ~= "." and image_file ~= "..") then
					imgs_count = imgs_count + 1
					-- print(img_file)
				end
			end
		end
	end
	print("Loading " .. type_str .. " data set (" .. imgs_count .. " images)")

	for sub_folder in lfs.dir(type_str .. "_set") do
		if (sub_folder ~= "." and sub_folder ~= "..") then
			label = tonumber(sub_folder)
			for img_filename in lfs.dir(type_str .. "_set/" .. sub_folder) do
				if (img_filename ~= "." and img_filename ~= "..") then
					img_filepath = type_str .. "_set/" .. sub_folder .. "/" .. img_filename
					image_file = image.load(img_filepath, 3, 'double')

					--print(">>>>>>>>> " .. image_file:dim())
					--print(".........." .. image_file:size(1) .. " " .. image_file:size(2) .. " " .. image_file:size(3))					

					--[[for i = 1,img_size do
						for j = 1,img_size do
							if image_file[1][i][j] > 0.5 then
								image_file[1][i][j] = 255
							else
								image_file[1][i][j] = 0
							end
						end
					end]]

					--local img = torch.DoubleTensor(3, img_size, img_size):fill(255)
					local img = image_file
					--img = ( img - 123 ) / 123
					--local ori_img = torch.ByteTensor(1, img_size, img_size):fill(255)

					type_data[type_idx] = {img, label, img_filename}

					type_idx = type_idx + 1
				end
			end
		end
	end

	for i = 1,table.getn(type_data) do
		type_idx_ary[i] = i
	end

	print("Finish loading " .. type_str .. " data set.")
end

function load_training_data()
	ori_imgs_train = { }
	imgs_train = { }
	labels_train = { }
	train_data = { }
	train_idx_ary = { }

	ori_imgs_type = ori_imgs_train
	imgs_type = imgs_train
	labels_type = labels_train
	type_data = train_data
	type_idx_ary = train_idx_ary
	type_str = "training"

	load_data()

	train_idx = 1
end

function load_test_data()
	ori_imgs_test = { }
	imgs_test = { }
	labels_test = { }
	test_data = { }
	test_idx_ary = { }

	ori_imgs_type = ori_imgs_test
	imgs_type = imgs_test
	labels_type = labels_test
	type_data = test_data
	type_idx_ary = test_idx_ary
	type_str = "test"

	load_data()

	test_idx = 1
end

function load_validate_data()
	ori_imgs_validate = { }
	imgs_validate = { }
	labels_validate = { }
	validate_data = { }
	validate_idx_ary = { }

	ori_imgs_type = ori_imgs_validate
	imgs_type = imgs_validate
	labels_type = labels_validate
	type_data = validate_data
	type_idx_ary = validate_idx_ary
	type_str = "validate"

	load_data()

	validate_idx = 1
end

function nextSample()
	epoch = epoch or 0

	-- whether goto next epoch
	if (train_idx == 1) then
		epoch = epoch + 1
		shuffle(train_idx_ary)
	end

	local train_ele = train_data[train_idx_ary[train_idx]]
	local img = train_ele[1]
	local target = train_ele[2]

	train_idx = (train_idx == table.getn(train_data)) and 1 or (train_idx + 1)

	return img, target
end