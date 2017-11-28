var express = require('express');
var router = express.Router();
var AWS = require('aws-sdk');
var fs=require('fs');
var request = require('request');
var multer  = require('multer')
var upload = multer().single('avatar')



var bucketName = 'advanced-data-analytics';
//var keyName = 'hello_world.txt';
/* GET users listing. */
router.get('/', function(req, res, next) {
  //res.send('respond with a resource');
    res.sendFile("hello.html",{"root": "./views/"})
});

router.post('/upload',function (req,res,next) {
    upload(req, res, function (err) {
        if (err) {
            // An error occurred when uploading
            console.log(err)
            res.send("Failed to uploaded package.");
        }
        var base64data = new Buffer(req.file.buffer , 'binary');
        var s3 = new AWS.S3();
        s3.putObject({
            Bucket: bucketName,
            Key: req.file.originalname,
            Body: base64data,
            ContentType:req.file.mimetype,
            ACL: 'public-read'
        },function (resp) {
            console.log(resp);
            console.log('Successfully uploaded package.');
            req.session.filename=req.file.originalname;
            console.log(req.session.filename);
            res.send(req.file.originalname);
        });
    });

});

var evaluateModel=function (endpoint) {
    return new Promise(function (resolve,reject) {
        var options = {
            uri: 'https://yvrq6r0cb8.execute-api.us-east-2.amazonaws.com/test/objectdetection',
            method: 'POST',
            json: {
                "url":endpoint
            }
        };
        request(options, function (error, response, body) {
            //console.log(response);
            if (!error && response.statusCode == 200) {
                resolve(response);
            }
            else{
                reject(error);
            }
        });
    });
}

router.post('/evaluate',function (req,res,next) {
    var baseUrl='https://s3.amazonaws.com/advanced-data-analytics/';

    var url=baseUrl+req.session.filename;
    console.log(url);
    evaluateModel(url).then(function (result) {
        res.send("Success");
    }).catch(function (error) {
       res.send('Fail to preprocess image,try later');
    });
});
module.exports = router;
