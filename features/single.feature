Feature: Single Image

    Scenario: Test making no changes to single image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose file "tmp/test_unchanged.png" from "saveDialog"
        Then "tmp/test_unchanged.png" and "test.jpg" should be equal
  
    Scenario: Test processing single image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I set "sharpen1" to "0.150"
        And I set "sharpen2" to "0.100"
        And I set "denoise1" to "0.5"
        And I set "deconvolveGaussianDiameter" to "2"
        And I async press "saveButton"
        And I choose file "tmp/test.png" from "saveDialog"
        Then "tmp/test.png" and "reference/test.png" should be equal
        
    Scenario: Test processing 16bit image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose file "16bit.png" from "openDialog"
        And I wait until active tab is "Process"
        And I set "sharpen1" to "0.600"
        And I set "sharpen2" to "0.300"
        And I set "gammaAdjust" to "85"
        And I set "valueAdjust" to "200"
        And I async press "saveButton"
        And I choose file "tmp/16bit.png" from "saveDialog"
        Then "tmp/16bit.png" and "reference/16bit.png" should be equal
