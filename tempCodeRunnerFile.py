while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if not ret or not ret1:
        print("Cannot receive frame")
        break

    img0 = cv2.flip(cv2.resize(frame, (768, 480)), 1)
    img1 = cv2.flip(cv2.resize(frame1, (768, 480)), 1)

    if frame_count % frame_interval == 0:
        # 在主循环中调用
        class_name, confidence_score = process_frame(frame, class_names, model)
        class_name1, confidence_score1 = process_frame(frame1, class_names, model)

        # 处理类别1的检测
        if class_name == 'Class 1':
            if confidence_score < 1:
                if not start_time_low_confidence:
                    start_time_low_confidence = time.time()
                elif (time.time() - start_time_low_confidence) > 3:
                    print("Class 1 confidence less than 100% for over 3 seconds, stopping emotion analysis.")
                    break
            else:
                start_time_low_confidence = None
            if not class_1_detected:
                class_1_detected = True
                start_time_1 = time.time()
            class_2_detected = False
        elif class_name == 'Class 2':
            if not class_2_detected:
                class_2_detected = True
                start_time_2 = time.time()
            class_1_detected = False
        else:
            class_1_detected = False
            class_2_detected = False
            start_time_1 = None
            start_time_2 = None

        # 在检测到类别1后3秒开始进行情绪、年龄和性别分析
        if class_1_detected and start_time_1 and (time.time() - start_time_1) > 3:
            try:
                analyze = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
                emotion = analyze[0]['dominant_emotion']
                age = round(analyze[0]['age'])
                gender_prob = analyze[0]['gender']
                gender = max(gender_prob, key=gender_prob.get)
                gender_confidence = round(gender_prob[gender], 2)
                emotions_over_time.append(emotion)
                ages_over_time.append(age)
                genders_over_time.append((gender, gender_confidence))
                previous_results.update({
                    'class_name': class_name,
                    'confidence_score': np.round(confidence_score * 100, 2),
                    'emotion': emotion,
                    'age': age,
                    'gender': gender,
                    'gender_confidence': gender_confidence
                })
                img0 = putText(img0, f"{class_name}, Confidence: {np.round(confidence_score * 100, 2)}%", 10, 30)
                img0 = putText(img0, f"Emotion: {emotion}", 10, 70)
                img0 = putText(img0, f"Age: {age}", 10, 110)
                img0 = putText(img0, f"Gender: {gender} {gender_confidence}%", 10, 150)
            except Exception as e:
                print("Error in emotion detection:", e)

            try:
                analyze1 = DeepFace.analyze(frame1, actions=['emotion', 'age', 'gender'], enforce_detection=False)
                emotion1 = analyze1[0]['dominant_emotion']
                age1 = round(analyze1[0]['age'])
                gender_prob1 = analyze1[0]['gender']
                gender1 = max(gender_prob1, key=gender_prob1.get)
                gender_confidence1 = round(gender_prob1[gender1], 2)
                emotions_over_time1.append(emotion1)
                ages_over_time1.append(age1)
                genders_over_time1.append((gender1, gender_confidence1))
                previous_results.update({
                    'class_name1': class_name1,
                    'confidence_score1': np.round(confidence_score1 * 100, 2),
                    'emotion1': emotion1,
                    'age1': age1,
                    'gender1': gender1,
                    'gender_confidence1': gender_confidence1
                })
                img1 = putText(img1, f"{class_name1}, Confidence: {np.round(confidence_score1 * 100, 2)}%", 10, 30)
                img1 = putText(img1, f"Emotion: {emotion1}", 10, 70)
                img1 = putText(img1, f"Age: {age1}", 10, 110)
                img1 = putText(img1, f"Gender: {gender1} {gender_confidence1}%", 10, 150)

            except Exception as e1:
                print("Error in emotion detection:", e1)

    else:
        # 在不处理帧的情况下显示上一次的结果
        img0 = putText(img0, f"{previous_results['class_name']}, Confidence: {previous_results['confidence_score']}%", 10, 30)
        img0 = putText(img0, f"Emotion: {previous_results['emotion']}", 10, 70)
        img0 = putText(img0, f"Age: {previous_results['age']}", 10, 110)
        img0 = putText(img0, f"Gender: {previous_results['gender']} {previous_results['gender_confidence']}%", 10, 150)
        
        img1 = putText(img1, f"{previous_results['class_name1']}, Confidence: {previous_results['confidence_score1']}%", 10, 30)
        img1 = putText(img1, f"Emotion: {previous_results['emotion1']}", 10, 70)
        img1 = putText(img1, f"Age: {previous_results['age1']}", 10, 110)
        img1 = putText(img1, f"Gender: {previous_results['gender1']} {previous_results['gender_confidence1']}%", 10, 150)

    # 如果类别2连续检测超过3秒则退出循环
    if class_2_detected and start_time_2 and (time.time() - start_time_2) > 3:
        print("Class 2 detected for more than 3 seconds, stopping emotion analysis.")
        break

    # 显示摄像头图像
    cv2.imshow('camera0', img0)
    cv2.imshow('camera1', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1