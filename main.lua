require 'torch'
require 'nn'
require 'optim'

require 'data'
require 'util'
require 'model'


model()

-- Criterion
criterion = nn.ClassNLLCriterion()

-- Prepare the data
load_training_data()
load_test_data()

x, dl_dx = m:getParameters()

feval = function(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end
	-- get next sample in the training set
	input, target = nextSample()
	-- reset gradients
	dl_dx:zero()
	-- evaluate the loss function and its derivative wrt x
	local pred = m:forward(input)
	local loss_x = criterion:forward(pred, target)
	m:backward(input, criterion:backward(pred, target))
	return loss_x, dl_dx
end

-- sgd parameters
sgd_params = {
	learningRate = 3e-5,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0.9
}

loss_ary = { }
test_err_rate = { }
train_err_rate = { }
loss_epoch = { }

evalCounter = 0
function train(time)
	last_epoch = epoch or 0
	star_num = star_num or 0
	line_star = 80
	for i =1,time do
		evalCounter = evalCounter + 1
		_, fs = optim.sgd(feval, x, sgd_params)
		if (i % 100 == 0) then
			if (last_epoch ~= epoch) then
				star_num = 0
				io.write("Epoch " .. epoch .. ": ")
				last_epoch = epoch
			end
			local percent = math.floor((evalCounter % table.getn(train_data)) / table.getn(train_data) * 100)
			local cur_star_num = math.floor(percent / (100 / line_star))
			for j = star_num+1,cur_star_num do
				io.write("=")
			end
			star_num = cur_star_num
			io.flush()
		end
		loss_ary[evalCounter] = fs[1]
	end
end

function calTrainErrRate()
	type_data = train_data
	type_str = "train"
	return calDataErrRate()
end

function calTestErrRate()
	type_data = test_data
	type_str = "test"
	return calDataErrRate()
end

function calDataErrRate()
	local errNum = 0
	print("Error rate on " .. type_str .. " set (data size: " .. table.getn(type_data) .. ").")
	local wrong_num = 0
	for i = 1,table.getn(type_data) do


		local ele = type_data[i]
		local input = ele[1]
		local target = ele[2]

		-- pedict
		local pred = m:forward(input)
		local m_t, m_i = torch.max(pred, 1)

		if (m_i[1] ~= target) then
			wrong_num = wrong_num + 1
		end
	end
	print("Error rate: " .. wrong_num / table.getn(type_data) .. ". Wrong number: " .. wrong_num)
	return wrong_num / table.getn(type_data)
end

function train_epoch(epoch_num)
	for e = 1, epoch_num do
		nClock = os.clock() 
		train(table.getn(train_data))
		local elapse = torch.round((os.clock() - nClock) * 10) / 10

		for j = star_num + 1, line_star do
			io.write("=")
		end
		io.flush()
		local loss_tensor = torch.Tensor(loss_ary)
		local num = table.getn(train_data)
		local loss_cur_epoch = loss_tensor:sub((last_epoch - 1) * num + 1, last_epoch * num):mean()
		io.write(". Ave loss: " .. loss_cur_epoch .. ".")
		loss_epoch[epoch] = loss_cur_epoch
		io.write(" Execution time: " .. elapse .. "s.")
		io.write("\n")

		-- test on the training set and test set
		train_err_rate[epoch] = calTrainErrRate()
		-- test_err_rate[epoch] = calTestErrRate()

		-- save the model file
		torch.save("models/" .. epoch .. ".mdl", m)
	end
end

train_epoch(10)