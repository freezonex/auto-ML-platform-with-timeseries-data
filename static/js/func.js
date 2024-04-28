$(document).ready(function(){
        // Function to update the image source and display the analysis image
        function updateAnalysisImage(imageUrl) {
            $('#analysis-image').hide();
            $('#analysis-image').on('load', function() {
                $(this).show();
            }).on('error', function() {
                toggleModal('Learning Mode Error','Failed to load image, please try again.');
                $(this).hide();
            });
            $('#analysis-image').attr('src', imageUrl);
        }
        function updateResultImage(imageUrl) {
            console.log("Function called with URL: ", imageUrl);
            $('#result-image').hide();
            $('#result-image').on('load', function() {
                $(this).show();
            }).on('error', function() {
                toggleModal('Learning Mode Error','Failed to load image, please try again.');
                $(this).hide();
            });
            $('#result-image').attr('src', imageUrl);
        }


        $("form").submit(function(event){
            event.preventDefault();  // 阻止表单默认提交
            var formData = new FormData(this);
            $.ajax({
                url: '/upload',
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(response) {
                    $("#message").text(response.message).show();  // 显示消息
                    setTimeout(function() { $("#message").hide(); }, 3000); // 3秒后隐藏消息
                    $("#display-btn").show(); // 显示"显示数据"按钮
                    $("#analyze-btn").show(); // 显示"开始数据分析"按钮
                },
                error: function() {
                    $("#message").text("Upload failed").show();
                    setTimeout(function() { $("#message").hide(); }, 3000);
                }
            });
        });

        $("#display-btn").click(function(){
            $.ajax({
                url: "/display-data",
                type: "GET",
                success: function(response) {
                    $("#data-display1").html(response).show();
                    $("#data-display").html(response).show();
                },
                error: function() {
                    $("#data-display").text("Error loading data.").show();
                }
            });
        });

        $("#analyze-btn").click(function(){
            $.ajax({
                url: "/pre-analyze",
                type: "GET",
                success: function(response) {
                    toggleModal('Learning Mode Info',response.message);
                    $("#learning-options").show(); // 显示学习类型选项
                }
            });
        });

        $("#unsupervised-btn").click(function(){
            // Handle unsupervised learning choice
            toggleModal('Learning Mode Error',"Unsupervised learning is currently not supported.");
        });

        $("#supervised-btn").click(function(){
            // Handle supervised learning choice
            toggleModal('Learning Mode Info',"Please proceed with supervised learning.");
            $("#supervised-options").show(); // 显示学习类型选项
        });

        $("#submit-supervised-btn").click(function(){
            var labelName = $("#label-input").val();
            var excludeInput = $("#exclude-features-input").val();
            var excludedFeatures = excludeInput ? excludeInput.split(",").map(function(item) {
                return item.trim();
            }).filter(function(item) { return item !== ""; }) : []; // 过滤掉任何空字符串

            $.ajax({
                url: "/set-supervised-options",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({label: labelName, excludedFeatures: excludedFeatures}),
                success: function(response) {
                    toggleModal('Learning Mode info',response.message);
                    $("#visualization").show();
                },
                error: function() {
                    toggleModal('Learning Mode Error',"Error setting supervised options.");
                }
            });
        });
        // Show analysis options when the visualization button is clicked
        $("#visualization").click(function(){
            $("#analysis-options").show(); // Show the buttons for different analyses
        });

        // Click handler for histogram analysis
         $("#histogram-analysis").click(function() {
            var d = new Date(); // To prevent image caching
            updateAnalysisImage("/generate_histogram?" + d.getTime());
        });

        // Click handler for scatter analysis
        $("#scatter-analysis").click(function() {
            var d = new Date(); // To prevent image caching
            updateAnalysisImage("/generate_scatter?" + d.getTime());
        });

        // Click handler for correlation analysis
        $("#correlation-analysis").click(function() {
            var d = new Date(); // To prevent image caching
            updateAnalysisImage("/generate_correlation?" + d.getTime());
        });
        $("#start-ml").click(function() {
            $("#analysis-image").hide();  // Hide the image
            $("#ml-options").show();  // Show options for ML type
        });


        // Handlers for choosing ML type and sending data to the server
        $("#classification, #regression").click(function() {
            var mode = $(this).attr('id');
            $.ajax({
                url: '/start_ml',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({mode: mode}),
                success: function(response) {
                    // Display the summary
                    $("#summary-label").text(response.label);
                    $("#summary-excluded-features").text(response.excluded_features.join(', '));
                    $("#summary-task-type").text(response.mode);
                    $("#summary").show();
                    toggleModal('Learning Mode Irror','Model training started in ' + mode + ' mode.');
                },
                error: function() {
                    toggleModal('Learning Mode Error','Error starting machine learning.');
                }
            });
        });
        $("#confirm-training").click(function() {
            var mode = $("#summary-task-type").text();  // Retrieve the mode from the summary
            $.ajax({
                url: '/confirm_training',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({mode: mode}),
                success: function(response) {
                    toggleModal('Learning Mode Info','Training completed successfully!');
                    $("#start-testing").show();
                },
                error: function() {
                    toggleModal('Learning Mode Error','Error during training.');
                }
            });
        });
        $("#start-testing").click(function() {
            $("#test-data-upload").show();  // Show the test data upload section
        });

        $("#test-data-form").submit(function(event) {
            event.preventDefault();  // Prevent the default form submission
            var formData = new FormData(this);
            $.ajax({
                url: '/upload-test-data',
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(response) {
                    $("#message1").text(response.message).show();  // 显示消息
                    setTimeout(function() {
                        $("#message1").hide();
                        $("#threshold-input").val('');  // Clear and show the threshold input
                        $("#threshold-input").show();
                        $("#show-results").show();
                    }, 3000);

                },
                error: function() {
                    $("#test-summary").text("Test data upload failed.").show();
                }
            });
        });

        $(document).ready(function() {
            $("#show-results").click(function() {
                var threshold = $("#threshold-input").val(); // Get the optional threshold value
                $(this).hide();  // Optionally hide the button after clicking
                $.ajax({
                    url: '/evaluate',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ threshold: threshold }),
                    success: function(response) {
                        $("#result-buttons").show();  // Show result buttons on successful evaluation
                        $("#evaluation-message").text(response.message).show();  // Ensure this element exists
                    },
                    error: function() {
                        $("#evaluation-message").text("Evaluation failed. Please try again.").show();
                    }
                });
            });
        });

        // Event handlers for result buttons
        $("#scatter-plot").click(function() {
            updateResultImage('/static/result_images/scatter.png'); // Path to the scatter plot image
        });

        $("#f1-score").click(function() {
            updateResultImage('/static/result_images/confusion_matrix.png'); // Path to the confusion matrix image
        });

        $("#feature-analysis").click(function() {
            updateResultImage('/static/result_images/importance.png'); // Path to the feature importance image
        });

        $('.file-input').change(function() {
            var fileName = $(this).val().split('\\').pop();
            $('#file-name-display').text(fileName);
        });
//
        $('.file-input2').change(function() {
            var fileName = $(this).val().split('\\').pop();
            $('#file-name-display2').text(fileName);
        });
//        $('#display-btn').removeClass('hidden'); // To show
//        $('#data-display').removeClass('hidden'); // To show
//
//        // And to hide them again
//        $('#display-btn').addClass('hidden'); // To hide
//        $('#data-display').addClass('hidden'); // To hide

    });

function toggleModal(title, message) {
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');

    modalTitle.textContent = title || 'Notification';  // Default title
    modalBody.textContent = message || 'Something happened!';  // Default message

    modal.classList.toggle('hidden');  // Toggle visibility of the modal
}


