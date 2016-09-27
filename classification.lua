require 'torch'
require 'nn'
require 'optim'

----------------------------------------
-- load test data for debug use
-- remove this part when release
function loadData()

	require 'data'
	load_test_data()

end

----------------------------------------
-- input : DoubleTensor img 1*28*28
-- output: int class
function predict(img)
	-- 0.85 at epoch = 3
	-- 1.00 at epoch = 5
	-- overfitting at epoch 10
	local epoch = 5

	local model = torch.load("models/" .. epoch .. ".mdl")

	-- pedict
	local pred = model:forward(img)
	local m_t, m_i = torch.max(pred, 1)

	return m_i[1]

end

----------------------------------------
--
--[[loadData()

testSize = table.getn(test_data)
errNum = 0

for i = 1, testSize do
	pred = predict(test_data[i][1])
	print("class: " .. test_data[i][2] .. " <-> " .. pred)

	if pred ~= test_data[i][2] then
		errNum = errNum + 1
	end
end

print("-ErrorNum: " .. errNum)
print("-TestSize: " .. testSize)
print("-Accuracy: " .. (testSize - errNum) / testSize)]]

function test()
	print("hello world!")
end
