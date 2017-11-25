var express = require('express');
var router = express.Router();
var AWS = require('aws-sdk');
var fs=require('fs');
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

router.post('/evaluate',function (req,res,next) {
    console.log(req.session.filename);
    res.send("Success");
});
module.exports = router;
