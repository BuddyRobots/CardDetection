require 'torch'
require 'nn'
require 'nnx'

function model()
	klass = 7
	m = nn.Sequential()

	-- Input 28*28*1
	-- First 14*14*32
	-- Second 7*7*64

	-- first convolution layer
	m:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- second convolution layer
	m:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- densely connected layer
	m:add(nn.Reshape(64 * 7 * 7))
	m:add(nn.Linear(64 * 7 * 7, klass))

	-- LogSoftMax
	m:add(nn.LogSoftMax())
end
