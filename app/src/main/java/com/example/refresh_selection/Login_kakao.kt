package com.example.refresh_selection

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatImageButton
import com.android.volley.Response
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import com.kakao.sdk.auth.model.OAuthToken
import com.kakao.sdk.user.UserApiClient
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*


class Login_kakao : AppCompatActivity() {
    @SuppressLint("WrongViewCast")
    val testUrl = "http:/ 3.19.218.150:8089//api/user/login"
    val fileName = "/data/data/com.example.refresh_selection/files"
    fun csvFirst(g_a: String) {
        val headerData = "Sex,Age,Month,time,day"
        val file = File(fileName, "gender_age.csv")

        if (file.exists() == true) {//파일이 있을시
            Log.d("Loginkakao", "파일 있었네")
            val bw = BufferedWriter(FileWriter(file, true))
            try {
                bw.write(g_a + "\n")
                bw.flush()
                bw.close()
                Log.d("fileCreate", "파일있을때 append")
            } catch (e: Exception) {
                Toast.makeText(this, "Error on writing file ${e.message}", Toast.LENGTH_LONG).show()
            }

        } else {//파일이 없을시
            val bw = BufferedWriter(FileWriter(file))
            try {
                bw.write("$headerData\n")
                Log.d("fileCreate", "파일 없어서 생성 create")
                bw.write(g_a + "\n")
                bw.flush()
                bw.close()
            } catch (e: Exception) {
                Toast.makeText(this, "Error on writing file ${e.message}", Toast.LENGTH_LONG).show()
            }
        }


    }
//    fun csvSave(data: ByteArray){
//        val file = File(fileName, "gender_age.csv")
//        val bw = BufferedWriter(FileWriter(file, true))
//        try {
//            bw.write("$parTime,"+data.joinToString(",")+"\n")
//            bw.flush()
//            bw.close()
//        }
//        catch (e: Exception) {
//            Toast.makeText(this, "Error on writing file ${e.message}", Toast.LENGTH_LONG).show()
//        }
//    }


    @RequiresApi(Build.VERSION_CODES.KITKAT)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.login_kakao)
        var gender_age = ""
        val nextIntent = Intent(this, MainActivity_travel::class.java)
        // 로그인 공통 callback 구성

        val callback: (OAuthToken?, Throwable?) -> Unit = { token, error ->
            if (error != null) {
                //Login Fail
                Log.e("login", "Kakao Login Failed :", error)
            } else if (token != null) {
                //Login Success
                // 사용자 정보 요청 (기본)
                UserApiClient.instance.me { user, error ->
                    if (error != null) {
                        Log.e("user info fail", "사용자 정보 요청 실패", error)
                    } else if (user != null) {
                        if (user.kakaoAccount?.gender.toString().equals("FEMALE")) {
                            gender_age += "F,"
                        } else {
                            gender_age += "M,"
                        }
                        var age=user.kakaoAccount?.ageRange.toString().split("_")
                        var rdn = (0..10).random()
                     when(age[1]){
                         "10" -> gender_age += "" + (10 + rdn)+","
                         "20" -> gender_age += "" + (20 + rdn)+","
                         "30" -> gender_age += "" + (30 + rdn)+","
                         "40" -> gender_age += "" + (40 + rdn)+","
                         "50" -> gender_age += "" + (50 + rdn)+","
                         "60" -> gender_age += "" + (60 + rdn)+","
                         "70" -> gender_age += "" + (70 + rdn)+","
                     }
                        val dt = Date()
                        val month = (dt.month + 1).toString() + "" //month+1 한 값이 월
                        val hour = dt.hours.toString() + ""
                        val sdf = SimpleDateFormat("EE") //요일
                        val day = sdf.format(dt).toString() //요일

                        gender_age += "$month,$hour,$day"
//                    SimpleDateFormat full_sdf = new SimpleDateFormat("yyyy-MM-dd, hh:mm:ss a");

                        csvFirst(gender_age)

                        nextIntent.putExtra("gender_age", gender_age)
                        Log.i(
                            "user info success", "사용자 정보 요청 성공" +
                                    "\n회원번호: ${user.id}" +
                                    "\n닉네임: ${user.kakaoAccount?.profile?.nickname}" +
                                    "\n프로필사진: ${user.kakaoAccount?.profile?.thumbnailImageUrl}" +
                                    "\n성별: ${user.kakaoAccount?.gender}" +
                                    "\n나이: ${user.kakaoAccount?.ageRange}" + "\n성별나이" + gender_age
                        )

                        val myJson = JSONObject()
                        myJson.put("UserNumber", user.id)//회원번호
                        myJson.put("Sex", user.kakaoAccount?.gender)//성별
                        myJson.put("AgeRange", user.kakaoAccount?.ageRange)//나이범위
                        myJson.put("Image", user.kakaoAccount?.profile?.thumbnailImageUrl)//프로필사진
                        myJson.put("NickName", user.kakaoAccount?.profile?.nickname)//닉네임
                        val requestBody = myJson.toString()
                        /* myJson에 아무 데이터도 put 하지 않았기 때문에 requestBody는 "{}" 이다 */

                        val testRequest = object : StringRequest(
                            Method.POST,
                            testUrl,
                            Response.Listener { response ->
                                println("서버 Response 수신: $response")//가져오기 성공
                            },
                            Response.ErrorListener { error ->
                                Log.d("ERROR", "서버 Response 가져오기 실패: $error")
                            }) {
                            override fun getBodyContentType(): String {
                                return "application/json; charset=utf-8"
                            }

                            override fun getBody(): ByteArray {
                                return requestBody.toByteArray()
                            }
                            /* getBodyContextType에서는 요청에 포함할 데이터 형식을 지정한다.
                             * getBody에서는 요청에 JSON이나 String이 아닌 ByteArray가 필요하므로, 타입을 변경한다. */
                        }

                        Volley.newRequestQueue(this).add(testRequest)


                    }
                }

                startActivity(nextIntent)
            }
        }
        var kakao_login_btn: AppCompatImageButton? = null
        kakao_login_btn = findViewById<AppCompatImageButton>(R.id.kakao_login_bt)
        kakao_login_btn.setOnClickListener {
            // 카카오톡이 설치되어 있으면 카카오톡으로 로그인, 아니면 카카오계정으로 로그인
            UserApiClient.instance.run {
                if (isKakaoTalkLoginAvailable(this@Login_kakao)) {
                    loginWithKakaoTalk(this@Login_kakao, callback = callback)
                } else {
                    loginWithKakaoAccount(this@Login_kakao, callback = callback)
                }
            }
        }


    }

    fun testVolley(context: Context, success: (Boolean) -> Unit) {

        val myJson = JSONObject()
//        myJson.put("UserNumber",user.id)//회원번호
//        myJson.put("Sex",user.kakaoAccount?.gender)//성별
//        myJson.put("AgeRange",user.kakaoAccount?.ageRange)//나이범위
//        myJson.put("Image",user.kakaoAccount?.profile?.thumbnailImageUrl)//프로필사진
//        myJson.put("NickName",user.kakaoAccount?.profile?.nickname)//닉네임
        val requestBody = myJson.toString()
        /* myJson에 아무 데이터도 put 하지 않았기 때문에 requestBody는 "{}" 이다 */

        val testRequest = object : StringRequest(
            Method.POST,
            testUrl,
            Response.Listener { response ->
                println("서버 Response 수신: $response")
                success(true)
            },
            Response.ErrorListener { error ->
                Log.d("ERROR", "서버 Response 가져오기 실패: $error")
                success(false)
            }) {
            override fun getBodyContentType(): String {
                return "application/json; charset=utf-8"
            }

            override fun getBody(): ByteArray {
                return requestBody.toByteArray()
            }
            /* getBodyContextType에서는 요청에 포함할 데이터 형식을 지정한다.
             * getBody에서는 요청에 JSON이나 String이 아닌 ByteArray가 필요하므로, 타입을 변경한다. */
        }

        Volley.newRequestQueue(context).add(testRequest)
    }

//    fun startMainActivity() {
//
//        startActivity(nextIntent)
//    }

}