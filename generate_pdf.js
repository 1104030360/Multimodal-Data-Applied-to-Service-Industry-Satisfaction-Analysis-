const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

(async () => {
    try {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        const pdf_folder = 'static/pdf';
        const pdf_filename = 'report1.pdf';

        // 确保目录存在
        const pdfDirectory = path.resolve(__dirname, pdf_folder);
        console.log('PDF Directory:', pdfDirectory);  // 打印目录路径

        if (!fs.existsSync(pdfDirectory)) {
            fs.mkdirSync(pdfDirectory, { recursive: true });
            console.log('Directory created:', pdfDirectory);  // 打印创建的目录路径
        }

        // 确保保存路径正确
        const pdfPath = path.join(pdfDirectory, pdf_filename);
        console.log('Saving PDF to:', pdfPath);  // 打印完整的保存路径

        // 加载你的 Flask 应用页面
        await page.goto('http://127.0.0.1:5000/', {waitUntil: 'networkidle2'});

        // 生成 PDF 文件
        await page.pdf({
            path: pdfPath,
            format: 'A4',
            printBackground: true,
        });

        console.log('PDF generated successfully at', pdfPath);  // 确认PDF生成成功并显示路径
        await browser.close();
    } catch (error) {
        console.error('Error generating PDF:', error);
    }
})();
