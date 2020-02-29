
/*
 * Init default PoseNet properties
 */
var posenetProperties = {
	architecture: 'ResNet50', // MobileNetV1 or ResNet50
	outputStride: 32, // 8, 16, 32 (the smaller the higher accuracy)
	inputResolution: 200, // 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800
	multiplier: 1, // 1, 0.75, 0.50 (MobileNetV1 only) (the smaller the lower accuracy)
	quantBytes: 1, // 1, 2, 4 (bytes per float) (the smaller the lower accuracy)
}

const config = {
	videoWidth: 600,
	videoHeight: 500,
	minPoseConfidence: 0.1,
    minPartConfidence: 0.6,
	// since images are being fed from a webcam, we want to feed in the
	// original image and then just flip the keypoints' x coordinates. If instead
	// we flip the image, then correcting left-right keypoint pairs requires a
	// permutation on all the keypoints.
	flipPoseHorizontal: false,
	color: 'aqua',
	boundingBoxColor: 'red',
	lineWidth: 2,
	mlpThreshold: 0.5
}


var realTime = true;
var requiredKeypoints = [
		'leftShoulder',
		// 'leftEar',
		'leftEye',
		'nose',
		'rightEye',
		// 'rightEar',
		'rightShoulder'
	]


/*
 * Main function that will run our code.
 * 
 * This is the main function of the slouch detector. It executes
 * both the posenet model and the MLP for detecting the
 * slouching.
 *
 */
async function run() {
	toggleLoadingUI(true);
	const net = await posenet.load(posenetProperties);
	toggleLoadingUI(false);

	let video = await loadVideo();

	const canvas = document.getElementById('output');
	canvas.width = config.videoWidth;
	canvas.height = config.videoHeight;

	const model = createModel();
	// tfvis.show.modelSummary({name: 'Model Summary'}, model);

	poseDetectionFrame(video, net, canvas, model);
}

/* BEGIN MLP */

/**
 * Predict if the user is slouching
 *
 * @param {array}	keypoints 	Array of keypoints predicted by posenet
 */
async function predictSlouching(keypoints, model) {

	keypoints = keypoints
		.filter(keypoint => (requiredKeypoints.includes(keypoint.part)))
		.filter(keypoint => (keypoint.score > config.minPartConfidence));

	var $alert = $('.alert');
	if (keypoints.length != requiredKeypoints.length) { // not enough body points detected
		var visibleParts = keypoints.map(keypoint => (keypoint.part));
		var part = requiredKeypoints.filter(requiredKeypoint => (visibleParts.indexOf(requiredKeypoint) === -1))[0];
		$alert.prop('class', 'alert alert-secondary');
		$alert.find('.alert-heading').html('Well, this is embarrassing but...')
		$alert.find('.alert-text').html(`It looks like I can't see your ${part}`);
		return;
	}

	const tensorData = convertToTensor(keypoints);
	const { inputTensor } = tensorData;

	// Train the model
	// await trainModel(model, inputs, labels);
	// console.log('Done Training');

	const pred = model.predict(inputTensor);
	const output = tf.sigmoid(pred).dataSync();
	if (output >= config.mlpThreshold) {
		$('.slouching-alert').addClass('active');
		$alert.prop('class', 'alert alert-warning');
		$alert.find('.alert-heading').html("Don't give up now")
		$alert.find('.alert-text').html('You can do it!');
	} else {
		$('.slouching-alert').removeClass('active');
		$alert.prop('class', 'alert');
		$alert.find('.alert-heading').html('You are doing great')
		$alert.find('.alert-text').html('Keep it up!');
	}

	console.log(output)
}


/**
 * Extract position from all the keypoints
 */
function convertToTensor(keypoints) {
	keypoints = keypoints
		.map(keypoint => ([
			keypoint.position.x,
			keypoint.position.y
		]))
		.flat();
	const inputs = [keypoints];

	const inputTensor = tf.tensor(inputs);

	// const normalizedInputs = inputTensor
	// 	.sub()
	// 	.div([config.videoWidth, config.videoHeight].sub([0, 0]));

	return { inputTensor }
}


/**
 * Create MLP slouching classifier. Takes the keypoints of posenet as input.
 */
function createModel() {
	const model = tf.sequential(); 

	inputDim = requiredKeypoints.length * 2;

	model.add(tf.layers.dense({inputDim: inputDim, units: 24, useBias: true}));
	model.add(tf.layers.dense({units: 1, useBias: true}));

	return model;
}


// async function trainModel(model, inputs, labels) {
//   // Prepare the model for training.  
//   model.compile({
//     optimizer: tf.train.adam(),
//     loss: tf.losses.meanSquaredError,
//     metrics: ['mse'],
//   });
  
//   const batchSize = 32;
//   const epochs = 50;
  
//   return await model.fit(inputs, labels, {
//     batchSize,
//     epochs,
//     shuffle: true,
//     callbacks: tfvis.show.fitCallbacks(
//       { name: 'Training Performance' },
//       ['loss', 'mse'], 
//       { height: 200, callbacks: ['onEpochEnd'] }
//     )
//   });
// }


/* END MLP */


/**
 * Toggles between the loading UI and the main canvas UI.
 *
 * @param {boolean}	showLoadingUI	Whether to show the loading UI
 * @param {string}	loadingDivId	id of the loading div element
 * @param {string}	mainDivId		id of the main div element
 */
function toggleLoadingUI(showLoadingUI,
						 loadingDivId = 'loading', mainDivId = 'main') {
	if (showLoadingUI) {
		document.getElementById(loadingDivId).style.display = 'block';
		document.getElementById(mainDivId).style.display = 'none';
	} else {
		document.getElementById(loadingDivId).style.display = 'none';
		document.getElementById(mainDivId).style.display = 'block';
	}
}


/**
 * Determines if the device is Android Mobile.
 */
function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}


/**
 * Determines if the device is iOS Mobile.
 */
function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}


/**
 * Determines if the device is Mobile.
 */
function isMobile() {
  return isAndroid() || isiOS();
}


/**
 * Asks for Camera permission to the browser and determines if the device
 * is Mobile
 *
 * @throws Error if GetUserMedia is not supported
 * @returns Promise for video object
 */
async function setupCamera() {
	if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
		throw new Error(
			'Browser API navigator.mediaDevices.getUserMedia not available');
	}

	const video = document.getElementById('video');
	video.width = config.videoWidth;
	video.height = config.videoHeight;

	const mobile = isMobile();
	const stream = await navigator.mediaDevices.getUserMedia({
		'audio': false,
		'video': {
			facingMode: 'user',
			width: mobile ? undefined : config.videoWidth,
			height: mobile ? undefined : config.videoHeight,
		},
	});
	video.srcObject = stream;

	return new Promise((resolve) => {
		video.onloadedmetadata = () => {
		resolve(video);
		};
	});
}


/**
 * Sets up the video and starts playing
 *
 * @throws Error if GetUserMedia is not supported
 * @returns video object
 */
async function loadVideo() {
	let video;
	try {
		video = await setupCamera();
	} catch (e) {
		var $alert = $('.alert');
		$alert.prop('class', 'alert alert-danger');
		$alert.find('.alert-heading').html('Video not supported')
		$alert.find('.alert-text').html('This browser does not support video capture,' +
			'or this device does not have a camera');
		throw e;
	}
	video.play();

	return video;
}


/**
 * Loads a new Image object given a path
 *
 * @param {string}	imgPath		Where to get the image from
 * @returns Promise for image object
 */
async function loadImage(imgPath) {
	const image = new Image();
	const promise = new Promise((resolve, reject) => {
		image.crossOrigin = '';
		image.onload = () => {
		  resolve(image);
		};
	});

	image.src = imgPath;
	return promise;
}

/* DRAW FUNCTIONS */

/**
 * Draws a line on a canvas, i.e. a joint
 */
function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
	ctx.beginPath();
	ctx.moveTo(ax * scale, ay * scale);
	ctx.lineTo(bx * scale, by * scale);
	ctx.lineWidth = config.lineWidth;
	ctx.strokeStyle = config.color;
	ctx.stroke();
}


function drawPoint(ctx, y, x, r, color) {
	ctx.beginPath();
	ctx.arc(x, y, r, 0, 2 * Math.PI);
	ctx.fillStyle = config.color;
	ctx.fill();
}


function toTuple({y, x}) {
	return [y, x];
}


/**
* Draws a pose skeleton by looking up all adjacent keypoints/joints
*/
function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
	const adjacentKeyPoints =
		posenet.getAdjacentKeyPoints(keypoints, minConfidence);

	adjacentKeyPoints.forEach((keypoints) => {
		drawSegment(
			toTuple(keypoints[0].position), toTuple(keypoints[1].position), config.color,
			scale, ctx);
	});
}


/**
* Draw pose keypoints onto a canvas
*/
function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
	for (let i = 0; i < keypoints.length; i++) {
		const keypoint = keypoints[i];
		if (keypoint.score < minConfidence)
			continue;
		const {y, x} = keypoint.position;
		drawPoint(ctx, y * scale, x * scale, 3, config.color);
	}
}


/**
* Draw the bounding box of a pose. For example, for a whole person standing
* in an image, the bounding box will begin at the nose and extend to one of
* ankles
*/
function drawBoundingBox(keypoints, ctx) {
	const boundingBox = posenet.getBoundingBox(keypoints);
	ctx.rect(
		boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
		boundingBox.maxY - boundingBox.minY);
	ctx.strokeStyle = config.boundingBoxColor;
	ctx.stroke();
}

/* END DRAW FUNCTIONS */


/**
 * Captures an image and runs the detection algorithm every
 * X seconds for efficiency
 *
 * @param {Object}	video			Video object for camera feed
 * @param {Object}	net				Initialized detection model
 * @param {Object}	canvas			Canvas to draw the result of the detection
 */
async function poseDetectionFrame(video, net, canvas, model) {
	/*
	 * Change to architecture
	 */
	// Important to purge variables and free up GPU memory when changing architecture
	// net.dispose
	// toggleLoadingUI(true);
	// net.await = posenet.load(<<new_properties>>)
	// toggleLoadingUI(false);
	//guiState.architecture = guiState.changeToArchitecture;
	//guiState.changeToArchitecture = null;

	/*
	 * Change to multiplier
	 */
	// Important to purge variables and free up GPU memory when changing architecture
	// net.dispose
	// toggleLoadingUI(true);
	// net.await = posenet.load(<<new_properties>>)
	// toggleLoadingUI(false);
	//guiState.multiplier = +guiState.changeToMultiplier;
	//guiState.changeToMultiplier = null;

	/*
	 * Change to outputStride
	 */
	// Important to purge variables and free up GPU memory when changing architecture
	// net.dispose
	// toggleLoadingUI(true);
	// net.await = posenet.load(<<new_properties>>)
	// toggleLoadingUI(false);
	//guiState.outputStride = +guiState.changeToOutputStride;
	//guiState.changeToOutputStride = null;

	/*
	 * Change to inputResolution
	 */
	// Important to purge variables and free up GPU memory when changing architecture
	// net.dispose
	// toggleLoadingUI(true);
	// net.await = posenet.load(<<new_properties>>)
	// toggleLoadingUI(false);
	//guiState.inputResolution = +guiState.changeToInputResolution;
	//guiState.changeToInputResolution = null;

	/*
	 * Change to quantBytes
	 */
	// Important to purge variables and free up GPU memory when changing architecture
	// net.dispose
	// toggleLoadingUI(true);
	// net.await = posenet.load(<<new_properties>>)
	// toggleLoadingUI(false);
	//guiState.quantBytes = +guiState.changeToQuantBytes;
	//guiState.changeToQuantBytes = null;

	let poses = [];

	let image = await loadImage(canvas.toDataURL());
	var input = tf.browser.fromPixels(image)

	const pose = await net.estimatePoses(input, {
		flipHorizontal: config.flipPoseHorizontal,
		decodingMethod: 'single-person'
		});
	poses = poses.concat(pose);

	const ctx = canvas.getContext('2d');
	ctx.clearRect(0, 0, config.videoWidth, config.videoHeight);

	// if (guiState.output.showVideo) {
		ctx.save();
		ctx.scale(-1, 1);
		ctx.translate(-config.videoWidth, 0);
		ctx.drawImage(video, 0, 0, config.videoWidth, config.videoHeight);
		ctx.restore();
	// }

	// For each pose (i.e. person) detected in an image, loop through the poses
	// and draw the resulting skeleton and keypoints if over certain confidence
	// scores
	poses.forEach(({score, keypoints}) => {
		if (score >= config.minPoseConfidence) {
			// if (guiState.output.showPoints)
				drawKeypoints(keypoints, config.minPartConfidence, ctx);
			// if (guiState.output.showSkeleton)
				drawSkeleton(keypoints, config.minPartConfidence, ctx);
			// if (guiState.output.showBoundingBox)
				// drawBoundingBox(keypoints, ctx);
		}
	});

	await predictSlouching(poses[0].keypoints, model)

	if (realTime)
		requestAnimationFrame(poseDetectionFrame.bind(null, video, net, canvas, model))
	else
		setTimeout(poseDetectionFrame.bind(null, video, net, canvas, model), 1000);
}



// Run main function
run();