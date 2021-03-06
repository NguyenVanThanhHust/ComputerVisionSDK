import sdk

mysdk = sdk.SDK()

error_code, face_detected, face_box_and_landmarks = mysdk.get_face_box_and_landmarks('../tests/images/face.jpg')

if error_code != sdk.ERRORCODE.NO_ERROR:
    print("There was an error!")
    quit()

if not face_detected:
    print("Unable to detected faces in image")
    quit()

print("Success, faces were detected")